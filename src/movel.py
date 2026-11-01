"""Uma `enlace2.Estacao` no celular: o laço de `saudacao.sessao`, sem PortAudio.

Roda dentro do app em `android/` (Chaquopy). O microfone e o alto-falante ficam
do lado Kotlin, em `Ponte`; aqui só entram e saem amostras, como `float32`
little-endian em `bytes`, porque converter um `float[]` elemento a elemento pela
ponte Java custaria o orçamento de CPU. Este módulo não importa `sounddevice`,
`painel` nem `volume`, e por isso não importa `saudacao`, que puxa os três.

O que difere de `saudacao.sessao`:

- *A caixa é digital.* A escada sai por `enlace.aplica_caixa`, como no simulado,
  com `CAIXA_POS` como fator (o maior é 1,00, então não satura o conversor --
  `CAIXA_SIM` passa de 1 e cortaria). O volume do sistema fica parado no que o
  app escolheu.
- *Não há ganho de microfone.* Os pedidos seguem o caminho de `vol=None` de
  `saudacao`: a estação fica sabendo que o microfone está no teto ou no piso.
- *Entrada perdida é contada, não reposta.* `saudacao.Lacunas` mora atrás do
  import de `sounddevice`. A `Ponte` informa as amostras que o `AudioRecord`
  andou além do que entregou, e o diário diz.

`volta` é um passo do laço, sem tempo de parede, para o teste rodar as duas
pontas no mesmo processo mais rápido que o tempo real.
"""

import base64
import json
import os
import queue
import time

import numpy as np

import enlace
import enlace2
import enlace_manual
import recording

FS = enlace.FS
BLOCO = 2048
ATRASO_AVISO = 2.0      # s de entrada na fila: acima disto a estação está atrás do tempo real
FATORES = enlace.CAIXA_POS


def manual_est(est):
    return getattr(est, 'manual', False)


class Sessao:
    """`ponte` é a `Ponte` do Kotlin: `le(ms)`, `pendentes()`, `toca(bytes)`,
    `cala()`, `latencia()`, `perdidas()`, `fonte()`, `linha(str)`, `estado(str)`,
    `recebido(caminho)`."""

    def __init__(self, papel, pasta, ponte, texto='', semente=None, manual=None):
        """`manual`: JSON dos campos de `enlace_manual.Params.de_dict`; None negocia."""
        self.papel, self.ponte = papel, ponte
        self.dados = texto.encode('utf-8')
        self.stem = os.path.join(pasta, time.strftime("%Y%m%d-%H%M%S") + f"-enlace-{papel}")
        self.diario = []
        self.ouvido = []
        self.comandos = queue.Queue()
        self.parar = False
        self.concluido = False
        self.motivo = 'parada no celular'
        self.lat = int(ponte.latencia())
        self.passo_ms = 0.0
        self.t_estado = 0.0
        self.acabou = False
        self.mostrados = 0
        self.t_aviso = 0.0
        if manual:
            self.est = enlace_manual.Estacao(papel, semente=semente, log=self.log,
                                             params=enlace_manual.Params.de_dict(json.loads(manual)))
        else:
            self.est = enlace2.Estacao(papel, semente=semente, log=self.log)
        self.est.enviar(self.dados)
        self.log(f"[movel] enlace2 versão {enlace2.VERSAO}, papel {papel}, "
                 f"latência declarada {self.lat / FS:.3f}s, "
                 f"{len(self.dados)} bytes a enviar")

    def log(self, s):
        self.diario.append(s)
        self.ponte.linha(s)

    def envia(self, texto):
        """Da interface: mais texto para a fila, de qualquer thread."""
        self.comandos.put(texto)

    def volta(self, x):
        """Um bloco ouvido; devolve a rajada a tocar ou None. Só protocolo."""
        est = self.est
        self.ouvido.append(x)
        est.ouvir(x)
        pendentes = int(self.ponte.pendentes())
        while pendentes > 0:
            b = self.ponte.le(0)
            if b is None:
                break
            y = np.frombuffer(bytes(b), dtype='<f4')
            self.ouvido.append(y)
            est.ouvir(y)
            pendentes = int(self.ponte.pendentes())
        tp = time.time()
        burst = est.passo()
        self.passo_ms = 1000 * (time.time() - tp)
        if burst is not None:
            est.desloca_saida(int(self.ponte.pendentes()) + self.lat)
            # Toda rajada sai no passo pedido: com marcas, trecho a trecho; sem
            # elas, no passo atual da estação, nunca no fator 1,00 do topo.
            if getattr(burst, 'caixa_marcas', None):
                burst = enlace.aplica_caixa(burst, FATORES, padrao=est.k)
            else:
                burst = burst * FATORES[est.k]
        est.pedidos_caixa.clear()
        for _, d, pico in est.pedidos_mic_passo:
            if d > 0:
                est.mic_no_teto = True
                self.log(f"[movel] microfone sem ajuste (pico ouvido {pico:.2f}); "
                         "quem sobe é a caixa do outro lado")
            else:
                est.mic_no_piso = True
                self.log(f"[movel] microfone sem ajuste (pico ouvido {pico:.2f})")
        est.pedidos_mic_passo.clear()
        est.pedidos_mic.clear()
        if (not manual_est(est) and not self.concluido and est.aberto and est.confirmado
                and not est.fila and est.pendente_tx is None):
            self.concluido = True
            self.log(f"[movel] tudo enviado e confirmado; recebidos {len(est.recebido)} bytes")
        return burst

    def espectro(self, bins=96, n=8192):
        """dB por faixa de 400 a 4600 Hz, a escala do mostrador; [] sem áudio."""
        if not self.ouvido:
            return []
        x = np.concatenate(self.ouvido[-(n // BLOCO + 1):])[-n:]
        if len(x) < n:
            return []
        p = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2 / n ** 2
        i = np.round(np.linspace(400, 4600, bins + 1) * n / FS).astype(int)
        return [round(float(10 * np.log10(p[a:b].max() + 1e-12)), 1) for a, b in zip(i[:-1], i[1:])]

    def publica(self):
        est = self.est
        fila = int(self.ponte.pendentes()) / FS
        # Só quando cresceu: a tela refaz a fita e a imagem a partir disto.
        dados = None
        if len(est.recebido) != self.mostrados:
            self.mostrados = len(est.recebido)
            dados = base64.b64encode(bytes(est.recebido)).decode('ascii')
        self.ponte.estado(json.dumps(dict(
            papel=self.papel, estado=getattr(est, 'estado', None), etapa=est.etapa,
            manual=manual_est(est), alerta=getattr(est, 'alerta', None),
            aberto=est.aberto, a_enviar=len(est.fila), recebidos=len(est.recebido),
            estacao_s=round(est.agora / FS, 1), fila_s=round(fila, 2),
            passo_ms=round(self.passo_ms, 1), perdidas_s=round(int(self.ponte.perdidas()) / FS, 2),
            rms=round(float(np.sqrt(np.mean(self.ouvido[-1] ** 2))), 4) if self.ouvido else None,
            enviados_total=len(self.dados), atividade=est.atividade(),
            contrato_tx=est.contrato_tx.descreve() if est.contrato_tx else None,
            contrato_rx=est.contrato_rx.descreve() if est.contrato_rx else None,
            percurso=est.percurso(parada=self.acabou),
            espectro=self.espectro(), dados=dados)))
        if fila > ATRASO_AVISO and time.time() - self.t_aviso > 10:
            self.t_aviso = time.time()
            self.log(f"[movel] ATENÇÃO: atrasado {fila:.1f} s em relação ao tempo real "
                     f"(passo {self.passo_ms:.0f} ms) -- a estação responde a sons velhos")

    def roda(self):
        """O laço até `parar` ou o FIM da estação. Bloqueia: é a thread de trabalho."""
        est = self.est
        self.log(f"[movel] entrada {self.ponte.fonte()}")
        try:
            while not self.parar:
                while not self.comandos.empty():
                    t = self.comandos.get().encode('utf-8') + b'\n'
                    est.enviar(t)
                    self.dados += t
                if (getattr(est, 'encerrada', False) and not est.agenda
                        and est.agora > est.livre_em + 3 * FS):
                    self.motivo = 'encerrada pelo FIM'
                    break
                if time.time() - self.t_estado > 0.25:
                    self.t_estado = time.time()
                    self.publica()
                b = self.ponte.le(500)
                if b is None:
                    continue
                burst = self.volta(np.frombuffer(bytes(b), dtype='<f4'))
                if burst is not None:
                    self.ponte.toca(np.clip(burst, -1, 1).astype('<f4').tobytes())
        except Exception as e:
            self.motivo = f'erro: {e!r}'
            raise
        finally:
            self.ponte.cala()
            self.fecha()

    def fecha(self):
        est = self.est
        self.acabou = True
        self.publica()
        if est.recebido:
            # O que chegou, como arquivo: a interface abre daqui.
            with open(self.stem + '-recebido.bin', 'wb') as f:
                f.write(bytes(est.recebido))
            self.ponte.recebido(self.stem + '-recebido.bin')
        self.log(f"[movel] {self.motivo}; recebidos {len(est.recebido)} bytes, "
                 f"entrada perdida {int(self.ponte.perdidas()) / FS:.2f}s")
        gravado = np.concatenate(self.ouvido) if self.ouvido else np.zeros(0)
        recording.save_as(self.stem, gravado, self.dados,
                          recorded=os.path.basename(self.stem)[:15], mode='enlace',
                          label=f'enlace-{self.papel}', papel=self.papel,
                          versao=enlace2.VERSAO, fonte=str(self.ponte.fonte()),
                          aberto=est.aberto, recebidos=len(est.recebido),
                          bits_errados=getattr(est, 'bits_errados', None),
                          bits_lidos=getattr(est, 'bits_lidos', None),
                          diario=self.diario, fim=self.motivo,
                          estacao_s=round(est.agora / FS, 1), simulado=None,
                          nota="sessão do enlace no celular")
        self.ponte.linha(f"[movel] gravada em {self.stem}")


def repete(wav, papel, saida):
    """Passa uma sessão gravada por uma estação, sem som, e escreve o diário em
    `saida`: o mesmo resultado do PC prova a pilha, e o fator é o orçamento de CPU."""
    x = recording.read(wav).astype(np.float32)
    d = []
    e = enlace2.Estacao(papel, semente=1, log=d.append)
    t0 = time.process_time()
    rajadas = 0
    for i in range(0, len(x), BLOCO):
        e.ouvir(x[i:i + BLOCO])
        if e.passo() is not None:
            rajadas += 1
            e.desloca_saida(2 * BLOCO)
    cpu = time.process_time() - t0
    d += [f"recebido {bytes(e.recebido)!r}", f"rajadas {rajadas} estado {e.estado}"]
    with open(saida, 'w', encoding='utf-8') as f:
        f.write('\n'.join(d) + '\n')
    return f"{cpu:.1f} s de CPU para {len(x) / FS:.0f} s de áudio: fator {cpu / (len(x) / FS):.3f}"
