package com.slouchpatrol

import android.content.Context
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.SamplerConfig
import java.io.File

/** Wraps Gemma 4 E4B running on the phone through LiteRT-LM. One fresh conversation per frame. */
class GemmaJudge(private val context: Context) {

    private var engine: Engine? = null
    var backendName = "not loaded"
        private set
    var modelName = "Gemma"
        private set

    fun findModel(): File? {
        val dirs = listOfNotNull(context.getExternalFilesDir(null), File("/data/local/tmp/llm"))
        return dirs.flatMap { it.listFiles()?.toList().orEmpty() }
            // The "-gpu" and "-web" builds have no vision encoder, so they can't see the camera.
            .filter { it.name.endsWith(".litertlm") && "-gpu" !in it.name && "-web" !in it.name }
            .maxByOrNull { it.length() }
    }

    /**
     * Loads the model. mode is "GPU", "CPU", or "MIXED" (text model on CPU, vision encoder on GPU).
     * Takes several seconds, so call off the main thread.
     */
    @OptIn(ExperimentalApi::class)
    fun load(model: File, maxImages: Int, mode: String) {
        // Speculative decoding loads an extra "verify" graph that pushes an 8 GB phone over its memory limit,
        // 280 tokens per image keeps enough detail to see a rounded back.
        ExperimentalFlags.enableSpeculativeDecoding = false
        ExperimentalFlags.visualTokenBudget = 280
        val gpu = { Backend.GPU() as Backend }
        val cpu = { Backend.CPU() as Backend }
        val attempts = when (mode.uppercase()) {
            "CPU" -> listOf(Triple("CPU", cpu, cpu))
            "MIXED" -> listOf(Triple("CPU+GPU vision", cpu, gpu))
            else -> listOf(Triple("GPU", gpu, gpu), Triple("CPU", cpu, cpu))
        }
        var lastError: Throwable? = null
        for ((name, backend, vision) in attempts) {
            try {
                val e = Engine(
                    EngineConfig(
                        modelPath = model.absolutePath,
                        backend = backend(),
                        visionBackend = vision(),
                        maxNumTokens = 2048,
                        maxNumImages = maxImages,
                        cacheDir = context.cacheDir.path,
                    )
                )
                e.initialize()
                engine = e
                backendName = name
                modelName = model.name.removeSuffix(".litertlm")
                return
            } catch (t: Throwable) {
                lastError = t
            }
        }
        throw IllegalStateException("Could not load ${model.name} on GPU or CPU", lastError)
    }

    /** Sends the frames (oldest first) plus the text turn and returns Gemma's raw reply. */
    fun judge(systemPrompt: String, jpegFrames: List<ByteArray>, userText: String): String {
        val e = engine ?: error("Model not loaded")
        val config = ConversationConfig(
            systemInstruction = Contents.of(systemPrompt),
            samplerConfig = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0),
            maxOutputToken = 48,
        )
        e.createConversation(config).use { conversation ->
            val parts: List<Content> = jpegFrames.map { Content.ImageBytes(it) } + Content.Text(userText)
            return conversation.sendMessage(Contents.of(parts)).toString()
        }
    }

    fun close() {
        engine?.close()
        engine = null
    }

    companion object {
        /** Used while the robot stands still and watches. */
        const val WATCH_PROMPT = """You are the vision module of a posture-enforcement robot. A phone camera is mounted on the robot's head.
Goal: rate how much the person in view, who is seated working at a desk or laptop, is slouching.
Look at the line from hip to shoulder to ear.
0 = upright: back straight and vertical, ears above the shoulders, chest open.
5 = clear slouch: upper back visibly rounded or leaning forward, head in front of the shoulders.
10 = extreme: hunched over, back curled, face close to the desk or screen.
Hard rules: if the head is at or below shoulder height, or the torso leans forward more than about 30 degrees from vertical, score at least 7.
Only score 0 to 2 if the back is clearly straight and vertical.
Also report where the person is horizontally in the NEWEST image ("left" third, "center" third, "right" third, or "none"),
and how far away they are: "reach" if they fill most of the frame height and are within arm's length, "near" if a few steps away, "far" otherwise.
Reply with exactly one line of JSON and nothing else:
{"slouch": integer 0 to 10, "person": "left" | "center" | "right" | "none", "distance": "far" | "near" | "reach"}"""

        /** Used while the robot walks to the person, one motion at a time. */
        const val APPROACH_PROMPT = """You are the navigation planner of a humanoid robot. A phone camera is mounted on the robot's head and you see what the robot sees.
Goal: reach the slouching person shown in the log and stop within arm's length, facing them.
Each turn you choose exactly one preloaded motion:
- "rotate_left": turn in place to the left by a small angle
- "rotate_right": turn in place to the right by a small angle
- "move_forward": walk a short step forward
- "arrived": the person is centered and within arm's length, so stop walking
Rules: if the person is in the left third, rotate_left; right third, rotate_right; centered and not within reach, move_forward;
centered and filling most of the frame height, arrived. If no person is visible, rotate toward the side where they were last seen in the log.
Reply with exactly one line of JSON and nothing else:
{"action": "rotate_left" | "rotate_right" | "move_forward" | "arrived", "person": "left" | "center" | "right" | "none", "distance": "far" | "near" | "reach"}"""
    }
}
