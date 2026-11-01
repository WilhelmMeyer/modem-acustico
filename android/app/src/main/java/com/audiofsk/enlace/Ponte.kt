package com.audiofsk.enlace

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTimestamp
import android.media.AudioTrack
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import android.os.Process
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * O microfone e o alto-falante, para `movel.Sessao`. As amostras cruzam para o
 * Python como float32 little-endian em ByteArray.
 *
 * A leitura do microfone nunca espera pelo Python: o relógio da estação são as
 * amostras que ela ouviu, então entrada perdida é o tempo andando para trás. A
 * saída fica sempre tocando (silêncio entre rajadas), para a latência ser a
 * mesma em toda rajada e rajada curta não ficar presa no buffer.
 */
class Ponte(
    contexto: Context,
    private val aoLinha: (String) -> Unit,
    private val aoEstado: (String) -> Unit,
    private val aoRecebido: (String) -> Unit,
) {
    private val FS = 48000
    private val BLOCO = 2048
    private val audio = contexto.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private val volumeOriginal = audio.getStreamVolume(AudioManager.STREAM_MUSIC)
    private val fila = LinkedBlockingQueue<ByteArray>()
    private val naFila = AtomicInteger(0)
    private val saida: AudioTrack
    private var entrada: AudioRecord? = null
    private var fonteUsada = "?"
    @Volatile private var rajada: FloatArray? = null
    @Volatile private var aberta = true
    @Volatile private var perdidasTotal = 0L
    private var leitor: Thread? = null
    private val escritor: Thread

    init {
        val maximo = audio.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
        val fixo = (maximo * 0.7).toInt()
        audio.setStreamVolume(AudioManager.STREAM_MUSIC, fixo, 0)
        aoLinha("[ponte] volume do sistema fixo em $fixo de $maximo; a escada da caixa é digital")
        val tam = AudioTrack.getMinBufferSize(FS, AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT)
        saida = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC).build())
            .setAudioFormat(AudioFormat.Builder().setSampleRate(FS)
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
            .setBufferSizeInBytes(maxOf(tam, 4 * BLOCO * 4))
            .setTransferMode(AudioTrack.MODE_STREAM).build()
        saida.play()
        escritor = Thread({ escreve() }, "saida").apply { start() }
    }

    private fun escreve() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
        val silencio = FloatArray(512)
        var atual: FloatArray? = null
        var pos = 0
        while (aberta) {
            val r = rajada
            if (r !== atual) { atual = r; pos = 0 }
            if (atual == null || pos >= atual.size) {
                saida.write(silencio, 0, silencio.size, AudioTrack.WRITE_BLOCKING)
                if (atual != null) { rajada = null; atual = null }
                continue
            }
            val n = minOf(512, atual.size - pos)
            saida.write(atual, pos, n, AudioTrack.WRITE_BLOCKING)
            pos += n
        }
    }

    /** Depois de a estação existir: quem ouve antes dela só enche a fila. */
    fun abre() {
        val tam = maxOf(FS * 4, AudioRecord.getMinBufferSize(FS, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_FLOAT))
        val semProcesso = audio.getProperty(AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED) == "true"
        val fonte = if (semProcesso) MediaRecorder.AudioSource.UNPROCESSED
                    else MediaRecorder.AudioSource.VOICE_RECOGNITION
        fonteUsada = if (semProcesso) "UNPROCESSED" else "VOICE_RECOGNITION (o aparelho não oferece UNPROCESSED)"
        val r = AudioRecord(fonte, FS, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_FLOAT, tam)
        check(r.state == AudioRecord.STATE_INITIALIZED) { "microfone não abriu" }
        desligaEfeitos(r.audioSessionId)
        entrada = r
        r.startRecording()
        leitor = Thread({ le(r, tam / 4) }, "entrada").apply { start() }
    }

    private fun desligaEfeitos(sessao: Int) {
        val feitos = mutableListOf<String>()
        if (AutomaticGainControl.isAvailable())
            AutomaticGainControl.create(sessao)?.let { it.enabled = false; feitos += "AGC" }
        if (NoiseSuppressor.isAvailable())
            NoiseSuppressor.create(sessao)?.let { it.enabled = false; feitos += "NS" }
        if (AcousticEchoCanceler.isAvailable())
            AcousticEchoCanceler.create(sessao)?.let { it.enabled = false; feitos += "AEC" }
        if (feitos.isNotEmpty()) aoLinha("[ponte] efeitos desligados: ${feitos.joinToString()}")
    }

    private fun le(r: AudioRecord, buffer: Int) {
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
        val b = FloatArray(BLOCO)
        val ts = AudioTimestamp()
        var lidas = 0L
        while (aberta) {
            val n = r.read(b, 0, BLOCO, AudioRecord.READ_BLOCKING)
            if (n <= 0) continue
            lidas += n
            val bytes = ByteBuffer.allocate(n * 4).order(ByteOrder.LITTLE_ENDIAN)
            bytes.asFloatBuffer().put(b, 0, n)
            naFila.addAndGet(n)
            fila.put(bytes.array())
            if (r.getTimestamp(ts, AudioTimestamp.TIMEBASE_MONOTONIC) == AudioRecord.SUCCESS) {
                val adiante = ts.framePosition - lidas - perdidasTotal
                if (adiante > buffer) perdidasTotal += adiante - buffer
            }
        }
    }

    // ---- chamados pelo Python ----
    fun le(ms: Int): ByteArray? {
        val b = if (ms <= 0) fila.poll() else fila.poll(ms.toLong(), TimeUnit.MILLISECONDS)
        if (b != null) naFila.addAndGet(-b.size / 4)
        return b
    }
    fun pendentes(): Int = naFila.get()
    fun toca(b: ByteArray) {
        val f = FloatArray(b.size / 4)
        ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(f)
        rajada = f
    }
    fun cala() { rajada = null }
    fun latencia(): Int = saida.bufferSizeInFrames
    fun perdidas(): Long = perdidasTotal
    fun fonte(): String = fonteUsada
    fun linha(s: String) = aoLinha(s)
    fun estado(s: String) = aoEstado(s)
    fun recebido(caminho: String) = aoRecebido(caminho)

    fun fecha() {
        aberta = false
        leitor?.join(1000)
        escritor.join(1000)
        entrada?.run { stop(); release() }
        saida.stop(); saida.release()
        audio.setStreamVolume(AudioManager.STREAM_MUSIC, volumeOriginal, 0)
    }
}
