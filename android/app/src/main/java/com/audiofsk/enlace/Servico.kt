package com.audiofsk.enlace

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager

/** Mantém a sessão viva fora da tela: o Android só deixa um app em segundo plano
 *  usar o microfone com um serviço assim, e com a notificação à vista. A estação
 *  continua na thread da `MainActivity`; aqui só a licença e o processador ligado. */
class Servico : Service() {
    private var trava: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(CANAL, "Sessão do enlace", NotificationManager.IMPORTANCE_LOW))
        val abre = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val aviso = Notification.Builder(this, CANAL)
            .setContentTitle("Enlace em sessão")
            .setContentText("Ouvindo e tocando pelo ar")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(abre)
            .setOngoing(true)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
            startForeground(1, aviso, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
        else
            startForeground(1, aviso)
        if (trava == null)
            trava = getSystemService(PowerManager::class.java)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "audiofsk:sessao").apply { acquire() }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        trava?.release()
        trava = null
        super.onDestroy()
    }

    private companion object { const val CANAL = "sessao" }
}
