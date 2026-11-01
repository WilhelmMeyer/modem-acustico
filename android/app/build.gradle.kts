import java.security.MessageDigest

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

// O protocolo é o da raiz do repositório, copiado a cada build e nunca editado
// aqui: uma cópia à mão acabaria falando outra coisa no fio que o PC.
val raiz = rootDir.parentFile
val arquivosProtocolo = listOf("fec.py", "modem.py", "xfer.py", "enlace.py", "enlace2.py", "enlace_manual.py")
val protocolo = layout.buildDirectory.dir("protocolo").get().asFile
copy {
    from(raiz) { include(arquivosProtocolo + listOf("recording.py", "movel.py")) }
    into(protocolo)
}
// As fontes do visor do PC: uma cópia à mão acabaria sendo outra identidade.
copy {
    from(File(raiz, "visor/fontes"))
    into(layout.projectDirectory.dir("src/main/assets/tela/fontes"))
}

// O que fala no fio: hash dos arquivos do protocolo. No PC, o mesmo número:
// cat fec.py modem.py xfer.py enlace.py enlace2.py enlace_manual.py | sha1sum | cut -c1-8
val hashProtocolo = MessageDigest.getInstance("SHA-1").run {
    arquivosProtocolo.forEach { update(File(raiz, it).readBytes()) }
    digest().joinToString("") { "%02x".format(it) }.take(8)
}

android {
    namespace = "com.audiofsk.enlace"
    compileSdk = 35
    defaultConfig {
        applicationId = "com.audiofsk.enlace"
        minSdk = 24             // AudioSource.UNPROCESSED e AudioRecord.getTimestamp
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += listOf("arm64-v8a") }
        buildConfigField("String", "PROTOCOLO", "\"$hashProtocolo\"")
    }
    buildFeatures { buildConfig = true }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

chaquopy {
    defaultConfig {
        version = "3.10"        // o scipy do Chaquopy não passa de 1.8.1 em Python 3.10
        buildPython(System.getenv("AUDIOFSK_PYTHON310") ?: "python3.10")
        pip { install("scipy==1.8.1") }
    }
    sourceSets { getByName("main") { srcDir(protocolo) } }
}
