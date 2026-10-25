package com.slouchpatrol

import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStream
import java.net.InetSocketAddress
import java.net.Socket

/**
 * Talks to the SONIC launcher over the local network: scripts/sonic_launcher.py (MuJoCo sim, port 5555)
 * or scripts/robot_launcher.py (real G1, port 5556). Both speak the same protocol.
 * Protocol: send "0".."5" plus newline, the launcher replies "0" once the trajectory finishes.
 * One connection is kept open and reused, as the launcher expects.
 */
object RobotLink {

    private val TRAJECTORIES = mapOf(
        "tap" to 0, "shake" to 1, "slap" to 2,
        "move_forward" to 3, "rotate_left" to 4, "rotate_right" to 5,
    )
    private val STATUS = mapOf(
        0 to "done", 1 to "unknown trajectory", 2 to "busy", 3 to "playback failed", 5 to "SONIC not running on the robot",
    )

    private var socket: Socket? = null
    private var reader: BufferedReader? = null
    private var out: OutputStream? = null
    private var connectedTo = ""

    /**
     * Plays one trajectory and blocks until the launcher reports it finished, so the next camera frame
     * shows where the robot ended up. Returns a status line starting with "200" on success.
     */
    @Synchronized
    fun run(address: String, motion: String): String {
        val idx = TRAJECTORIES[motion] ?: return "400 unknown motion $motion"
        var last = ""
        // Venue Wi-Fi can lag by a second or more, so retry the connection a few times before giving up.
        repeat(3) { attempt ->
            try {
                send(address, idx.toString())
                val code = readLine().toIntOrNull() ?: -1
                return if (code == 0) "200 $motion done" else "500 $motion: ${STATUS[code] ?: "status $code"}"
            } catch (t: Throwable) {
                disconnect()
                last = "error ${t.javaClass.simpleName}: ${t.message} (attempt ${attempt + 1})"
                Thread.sleep(500)
            }
        }
        return last
    }

    /** Asks the launcher to list its trajectories. */
    @Synchronized
    fun health(address: String): String = try {
        send(address, "?")
        generateSequence { readLine() }.takeWhile { it != "." }.joinToString("; ").let { "200 $it" }
    } catch (t: Throwable) {
        disconnect()
        "error ${t.javaClass.simpleName}: ${t.message}"
    }

    private fun send(address: String, text: String) {
        if (socket?.isConnected != true || connectedTo != address) connect(address)
        out!!.write("$text\n".toByteArray())
        out!!.flush()
    }

    private fun readLine(): String = reader!!.readLine()?.trim() ?: error("launcher closed the connection")

    private fun connect(address: String) {
        disconnect()
        val clean = address.removePrefix("http://").removePrefix("tcp://").trimEnd('/')
        val host = clean.substringBefore(':')
        val port = clean.substringAfter(':', "5555").toIntOrNull() ?: 5555
        val s = Socket()
        s.connect(InetSocketAddress(host, port), 6000)
        s.soTimeout = 60_000 // each trajectory is a fresh sim run: hold + clip + end hold
        socket = s
        reader = BufferedReader(InputStreamReader(s.getInputStream()))
        out = s.getOutputStream()
        connectedTo = address
    }

    private fun disconnect() {
        try {
            socket?.close()
        } catch (_: Throwable) {
        }
        socket = null
        reader = null
        out = null
    }
}
