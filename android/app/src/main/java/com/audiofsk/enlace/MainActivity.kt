package com.audiofsk.enlace

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.util.Base64
import android.view.WindowManager
import android.webkit.JavascriptInterface
import android.webkit.WebView
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONObject
import java.io.File
import kotlin.concurrent.thread

/** A tela é `assets/tela/index.html`, o mesmo aparelho do visor do PC; aqui só a ligação. */
class MainActivity : Activity() {
    private lateinit var web: WebView
    @Volatile private var sessao: PyObject? = null
    @Volatile private var ocupado = false

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(estado: Bundle?) {
        super.onCreate(estado)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.statusBarColor = Color.parseColor("#1b1a18")
        window.navigationBarColor = Color.parseColor("#1b1a18")
        if (!Python.isStarted()) Python.start(AndroidPlatform(this))
        val pede = listOf(Manifest.permission.RECORD_AUDIO, "android.permission.POST_NOTIFICATIONS")
            .filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (pede.isNotEmpty()) requestPermissions(pede.toTypedArray(), 1)
        web = WebView(this).apply {
            setBackgroundColor(Color.parseColor("#1b1a18"))
            fitsSystemWindows = true
            settings.javaScriptEnabled = true
            addJavascriptInterface(Controles(), "ponte")
            webViewClient = object : android.webkit.WebViewClient() {
                override fun onPageFinished(v: WebView?, url: String?) {
                    js("tela.protocolo(${JSONObject.quote(BuildConfig.PROTOCOLO)})")
                    linha("[app] gravações em ${pasta()}")
                }
            }
            loadUrl("file:///android_asset/tela/index.html")
        }
        setContentView(web)
    }

    private fun js(codigo: String) = runOnUiThread { web.evaluateJavascript(codigo, null) }
    private fun pasta(): File = File(getExternalFilesDir(null), "sessoes").apply { mkdirs() }
    private fun linha(s: String) = js("tela.linha(${JSONObject.quote(s)})")
    private fun mostra(j: String) = js("tela.estado($j)")
    private fun emSessao(ligada: Boolean) = js("tela.sessao($ligada)")

    /** O que chegou, aberto: imagem no papel; outro arquivo, só o resumo. */
    private fun abre(caminho: String) {
        val f = File(caminho)
        val b = BitmapFactory.decodeFile(caminho)
        if (b == null) {
            js("tela.arquivo(${JSONObject.quote("${f.length()} B em ${f.name}")})")
            return
        }
        val uri = "data:image/bmp;base64," + Base64.encodeToString(f.readBytes(), Base64.NO_WRAP)
        js("tela.imagem(${JSONObject.quote(uri)}, ${JSONObject.quote("${b.width} x ${b.height}, ${f.length()} B")})")
        linha("[app] imagem aberta: ${b.width}x${b.height}, de $caminho")
    }

    /** `manual`: JSON dos parâmetros escolhidos à mão; vazio, negocia pelo ar. */
    private fun inicia(papel: String, inicial: String, manual: String) {
        if (ocupado) return
        ocupado = true
        emSessao(true)
        startForegroundService(Intent(this, Servico::class.java))
        thread(name = "estacao") {
            var ponte: Ponte? = null
            try {
                ponte = Ponte(this, ::linha, ::mostra, ::abre)
                linha("[app] carregando a estação...")
                val s = Python.getInstance().getModule("movel")
                    .callAttr("Sessao", papel, pasta().path, ponte, inicial, null, manual.ifEmpty { null })
                ponte.abre()
                sessao = s
                s.callAttr("roda")
            } catch (e: Throwable) {
                linha("[app] ERRO: $e")
            } finally {
                sessao = null
                ponte?.fecha()
                stopService(Intent(this, Servico::class.java))
                emSessao(false)
                ocupado = false
            }
        }
    }

    /** `replay.wav` posto na pasta do app: o diário sai em `replay.txt`, para comparar com o PC. */
    private fun repete(papel: String) {
        val wav = File(getExternalFilesDir(null), "replay.wav")
        if (!wav.exists()) { linha("[app] ERRO: falta $wav"); return }
        if (ocupado) return
        ocupado = true
        thread(name = "repete") {
            try {
                val r = Python.getInstance().getModule("movel").callAttr("repete", wav.path, papel,
                    File(getExternalFilesDir(null), "replay.txt").path)
                linha("[app] repetição: $r")
            } catch (e: Throwable) {
                linha("[app] ERRO: $e")
            } finally {
                ocupado = false
            }
        }
    }

    inner class Controles {
        @JavascriptInterface fun iniciar(papel: String, texto: String, manual: String) = inicia(papel, texto, manual)
        @JavascriptInterface fun parar() { sessao?.put("parar", true) }
        @JavascriptInterface fun enviar(texto: String) { sessao?.callAttr("envia", texto) }
        @JavascriptInterface fun repetir(papel: String) = repete(papel)
    }
}
