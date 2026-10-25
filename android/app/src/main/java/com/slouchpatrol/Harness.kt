package com.slouchpatrol

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

data class Settings(
    val orinUrl: String = "192.168.7.23:5555",
    val imagesPerCall: Int = 1,
    val voteWindow: Int = 2,
    val voteThreshold: Int = 2,
    val ttsDelayMs: Long = 0,
    val extraRotation: Int = 0,
    val intervalMs: Long = 1000,
    val maxSteps: Int = 10,
    val settleMs: Long = 700,
    val backend: String = "GPU",
    val slouchThreshold: Int = 5,
    /** 0 = Gemma plans the walk step by step; N > 0 = walk forward exactly N times, then punish. */
    val fixedSteps: Int = 0,
    /** Gemma mode never takes more than this many forward steps, whatever Gemma says. */
    val maxForward: Int = 3,
    /** Who to watch, finishing the phrase "the person wearing ...". Everyone else is ignored. */
    val target: String = "a bright pink wig",
)

data class Observation(val t: Double, val score: Int, val slouching: Boolean, val person: String, val distance: String)

enum class State { LOADING, WATCHING, APPROACHING, PUNISHING, COOLDOWN, PAUSED }

/**
 * The loop around Gemma.
 * WATCHING: one frame a second, Gemma says slouching or not; a vote over recent answers decides.
 * APPROACHING: Gemma picks rotate_left / rotate_right / move_forward from each new frame, one motion
 *   at a time, until it says "arrived". A gate overrides choices that break the rules.
 * PUNISHING: tap on the first strike, shake on the second, slap on the third and after, plus a voice line.
 * COOLDOWN: wait for a manual Reset after the robot is put back on its mark.
 */
class Harness(
    private val gemma: GemmaJudge,
    private val latestFrame: () -> Bitmap?,
    private val speak: (String) -> Unit,
    private val onUpdate: (Snapshot) -> Unit,
    private val onModelReady: () -> Unit,
) {
    data class Snapshot(
        val state: State,
        val lastRaw: String,
        val latencyMs: Long,
        val log: String,
        val sentFrame: Bitmap?,
        val note: String,
        val strikes: Int,
    )

    @Volatile var settings = Settings()
    @Volatile var state = State.LOADING
        private set

    private val startMs = System.currentTimeMillis()
    private val history = ArrayDeque<Bitmap>()
    private val observations = ArrayDeque<Observation>()
    private val log = ArrayDeque<JSONObject>()
    private var lastRaw = ""
    private var latencyMs = 0L
    private var sentFrame: Bitmap? = null
    private var note = ""
    private var strikes = 0
    private var lastSeen = "center"
    private var loop: Job? = null
    private lateinit var scope: CoroutineScope

    // Everything that touches the harness state runs on this one thread, so no locking is needed.
    private val worker = Executors.newSingleThreadExecutor().asCoroutineDispatcher()

    fun start(scope: CoroutineScope) {
        this.scope = scope
        loop = scope.launch(worker) {
            val model = gemma.findModel()
            if (model == null) {
                publish(State.LOADING, "No .litertlm model found. Push it with adb (see README).")
                return@launch
            }
            publish(State.LOADING, "Loading ${model.name} (${model.length() / 1_000_000} MB)...")
            try {
                gemma.load(model, maxImages = MAX_IMAGES, mode = settings.backend)
            } catch (t: Throwable) {
                publish(State.LOADING, "Model load failed: ${t.message} / ${t.cause?.message}")
                return@launch
            }
            publish(State.WATCHING, "Model ready on ${gemma.backendName}")
            onModelReady()
            while (isActive) {
                val began = System.currentTimeMillis()
                if (state == State.WATCHING && watchTick()) approachAndPunish()
                val wait = settings.intervalMs - (System.currentTimeMillis() - began)
                delay(wait.coerceAtLeast(50))
            }
        }
    }

    /** One watch step. Returns true when the vote says it's time to go. */
    private fun watchTick(): Boolean {
        val frame = latestFrame() ?: return false
        val s = settings
        history.addLast(rotate(frame, s.extraRotation))
        while (history.size > s.imagesPerCall.coerceIn(1, MAX_IMAGES)) history.removeFirst()

        val raw = ask(GemmaJudge.watchPrompt(settings.target), history.toList(), watchTurn(history.size))
        val json = parse(raw) ?: return false.also { publish(state, "Could not parse Gemma's reply") }
        val score = json.optInt("slouch", if (json.optBoolean("slouching", false)) 10 else 0)
        val obs = Observation(now(), score, score >= s.slouchThreshold, person(json), distance(json))
        if (obs.person != "none") lastSeen = obs.person
        observations.addLast(obs)
        while (observations.size > s.voteWindow) observations.removeFirst()
        addLog(JSONObject().put("t", obs.t).put("slouch", obs.score).put("person", obs.person)
            .put("distance", obs.distance))

        Log.i(TAG, "watch slouch=${obs.score} person=${obs.person} distance=${obs.distance} raw=${raw.trim()}")
        val votes = observations.count { it.slouching && it.person != "none" }
        publish(state, "slouch ${obs.score}/10 (fires at ${s.slouchThreshold}) · votes $votes/${observations.size} (need ${s.voteThreshold})")
        return votes >= s.voteThreshold
    }

    private suspend fun approachAndPunish() {
        strikes++
        val punishment = PUNISHMENTS[(strikes - 1).coerceAtMost(PUNISHMENTS.lastIndex)]
        addLog(JSONObject().put("t", now()).put("event", "slouch confirmed").put("strike", strikes)
            .put("punishment", punishment.motion))
        publish(State.APPROACHING, "Strike $strikes: walking over to ${punishment.motion}")

        if (settings.fixedSteps > 0) {
            if (!walkFixed(settings.fixedSteps)) return
        } else if (!walkWithGemma()) {
            return
        }

        publish(State.PUNISHING, "${punishment.motion}!")
        scope.launch {
            delay(settings.ttsDelayMs)
            speak(punishment.line)
        }
        val reply = withContext(Dispatchers.IO) { RobotLink.run(settings.orinUrl, punishment.motion) }
        addLog(JSONObject().put("t", now()).put("event", punishment.motion).put("robot", reply))
        publish(State.COOLDOWN, "Done (strike $strikes). Put the robot back on its mark, then tap Reset.")
    }

    /** Fixed mode: a set number of forward steps with no Gemma calls, for a predictable stage demo. */
    private suspend fun walkFixed(steps: Int): Boolean {
        for (step in 1..steps) {
            if (state != State.APPROACHING) return false // Reset or Pause pressed mid-walk
            publish(State.APPROACHING, "Fixed step $step of $steps: move_forward")
            val reply = withContext(Dispatchers.IO) { RobotLink.run(settings.orinUrl, "move_forward") }
            addLog(JSONObject().put("t", now()).put("step", step).put("do", "move_forward").put("mode", "fixed"))
            if (!reply.startsWith("200")) {
                publish(State.COOLDOWN, "Robot error on move_forward: $reply. Tap Reset when fixed.")
                return false
            }
        }
        return state == State.APPROACHING
    }

    /** Gemma mode: one motion per frame until Gemma and the gate agree the person is within reach. */
    private suspend fun walkWithGemma(): Boolean {
        var arrived = false
        var forwardSoFar = 0
        var lastDistance = "far"
        for (step in 1..settings.maxSteps) {
            delay(settings.settleMs) // let the camera settle after the last motion
            if (state != State.APPROACHING) return false // Reset or Pause pressed mid-walk
            val frame = latestFrame()?.let { rotate(it, settings.extraRotation) } ?: continue
            val raw = ask(GemmaJudge.approachPrompt(settings.target), listOf(frame), approachTurn(step))
            val json = parse(raw)
            val seen = json?.let { person(it) } ?: "none"
            if (seen != "none") lastSeen = seen
            val chosen = json?.optString("action").orEmpty()
            val dist = json?.let { distance(it) } ?: "far"
            val action = gate(chosen, seen, dist, forwardSoFar, lastDistance)
            if (seen != "none") lastDistance = dist
            if (action == "move_forward") forwardSoFar++
            addLog(JSONObject().put("t", now()).put("step", step).put("gemma", chosen).put("do", action)
                .put("person", seen).put("distance", dist))
            Log.i(TAG, "step $step gemma=$chosen person=$seen distance=$dist -> $action (forward so far $forwardSoFar)")
            if (action == "arrived") {
                arrived = true
                break
            }
            publish(State.APPROACHING, "Step $step: $action")
            val reply = withContext(Dispatchers.IO) { RobotLink.run(settings.orinUrl, action) }
            if (!reply.startsWith("200")) {
                publish(State.COOLDOWN, "Robot error on $action: $reply. Tap Reset when fixed.")
                return false
            }
        }
        if (!arrived) {
            addLog(JSONObject().put("t", now()).put("event", "gave up").put("steps", settings.maxSteps))
            publish(State.COOLDOWN, "Didn't reach the person in ${settings.maxSteps} steps. Reposition, then Reset.")
        }
        return arrived
    }

    /**
     * The rule check between Gemma and the robot. Gemma proposes; this decides.
     * It errs toward stopping: Gemma's "arrived" is always trusted, "reach" always stops, losing sight of the
     * person after walking toward them means they're too close to see, and forward steps are capped.
     */
    private fun gate(chosen: String, person: String, distance: String, forwardSoFar: Int, lastDistance: String): String =
        when {
            chosen == "arrived" || distance == "reach" -> "arrived"
            forwardSoFar >= settings.maxForward -> "arrived"
            person == "none" && forwardSoFar > 0 && lastDistance != "far" -> "arrived"
            person == "none" -> if (lastSeen == "right") "rotate_right" else "rotate_left"
            chosen == "rotate_left" || chosen == "rotate_right" -> chosen
            person == "center" -> "move_forward"
            else -> "rotate_$person"
        }

    /** Plays one motion by hand, for testing the robot link. */
    fun testMotion(motion: String) {
        scope.launch(worker) {
            val reply = withContext(Dispatchers.IO) { RobotLink.run(settings.orinUrl, motion) }
            addLog(JSONObject().put("t", now()).put("event", "test").put("motion", motion).put("robot", reply))
            publish(state, "Test $motion: $reply")
        }
    }

    fun reset() {
        scope.launch(worker) {
            if (state == State.LOADING) return@launch
            observations.clear()
            history.clear()
            addLog(JSONObject().put("t", now()).put("event", "reset"))
            publish(State.WATCHING, "Watching · strikes $strikes")
        }
    }

    fun forgive() {
        scope.launch(worker) {
            strikes = 0
            addLog(JSONObject().put("t", now()).put("event", "strikes cleared"))
            publish(state, "Strikes cleared")
        }
    }

    fun togglePause() {
        scope.launch(worker) {
            when (state) {
                State.PAUSED -> publish(State.WATCHING, "Watching")
                State.WATCHING, State.APPROACHING -> publish(State.PAUSED, "Paused")
                else -> Unit
            }
        }
    }

    private fun ask(prompt: String, frames: List<Bitmap>, text: String): String {
        val jpegs = frames.map { toJpeg(it) }
        val began = System.currentTimeMillis()
        val raw = try {
            gemma.judge(prompt, jpegs, text)
        } catch (t: Throwable) {
            "error: ${t.message}"
        }
        latencyMs = System.currentTimeMillis() - began
        lastRaw = raw.trim()
        sentFrame = frames.last()
        return raw
    }

    private fun watchTurn(frameCount: Int): String {
        val frames = if (frameCount == 1) "The image above is the current camera frame."
        else "The $frameCount images above are recent camera frames, oldest first, about " +
            "${settings.intervalMs / 1000.0}s apart. Judge the NEWEST one; use the others only as context."
        return "$frames\nRecent log (oldest first): ${recentLog()}\nAnswer with the JSON line only."
    }

    private fun approachTurn(step: Int): String =
        "The image above is what the robot sees now, before step $step of at most ${settings.maxSteps}.\n" +
            "Recent log (oldest first): ${recentLog()}\nAnswer with the JSON line only."

    private fun recentLog() = JSONArray(log.toList().takeLast(5)).toString()

    private fun parse(raw: String): JSONObject? = try {
        JSONObject(Regex("\\{[^{}]*\\}").find(raw)!!.value)
    } catch (t: Throwable) {
        null
    }

    private fun person(json: JSONObject) = json.optString("person", "none").lowercase()
        .let { if (it in setOf("left", "center", "right")) it else "none" }

    private fun distance(json: JSONObject) = json.optString("distance", "far").lowercase()
        .let { if (it in setOf("far", "near", "reach")) it else "far" }

    private fun addLog(entry: JSONObject) {
        log.addLast(entry)
        while (log.size > 30) log.removeFirst()
    }

    private fun now() = (System.currentTimeMillis() - startMs) / 100 / 10.0

    private fun publish(newState: State, message: String) {
        state = newState
        note = message
        onUpdate(Snapshot(state, lastRaw, latencyMs, log.toList().takeLast(12).joinToString("\n"), sentFrame,
            note, strikes))
    }

    private fun rotate(src: Bitmap, degrees: Int): Bitmap {
        if (degrees % 360 == 0) return src
        val m = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    }

    private fun toJpeg(bitmap: Bitmap): ByteArray {
        val longSide = maxOf(bitmap.width, bitmap.height)
        val scaled = if (longSide > 768) {
            val f = 768f / longSide
            Bitmap.createScaledBitmap(bitmap, (bitmap.width * f).toInt(), (bitmap.height * f).toInt(), true)
        } else bitmap
        return ByteArrayOutputStream().use { out ->
            scaled.compress(Bitmap.CompressFormat.JPEG, 85, out)
            out.toByteArray()
        }
    }

    private data class Punishment(val motion: String, val line: String)

    companion object {
        private const val TAG = "SlouchPatrol"

        /** Kept small so the vision encoder and KV cache fit in an 8 GB phone's memory. */
        const val MAX_IMAGES = 5

        private val PUNISHMENTS = listOf(
            Punishment("tap", "Sit up straight."),
            Punishment("shake", "I said, sit up straight."),
            Punishment("slap", "You need to stop slouching."),
        )
    }
}
