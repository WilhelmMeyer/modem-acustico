"""Painel de uma sessão do enlace rodando em segundo plano.

Três coisas permitem acompanhar e parar de fora uma sessão de `saudacao.py`,
todas aqui e nenhuma em `enlace.py` (que não faz I/O):

* **Disco contínuo.** Cada bloco do microfone vai para `<stem>.parcial.f32`
  (float32 cru, mono, 48 kHz) e cada linha do diário para `<stem>.diario.txt`
  no momento em que acontece. Um processo morto à força deixa os dois; o
  `recupera` transforma o `.f32` no WAV de sempre.
* **Estado.** `<stem>.estado.json` é reescrito a cada segundo: fase, bytes,
  contratos e, sobretudo, se a estação acompanha o tempo real (`atraso_s`:
  relógio de parede menos relógio da estação, descontada a pausa; `fila_s`:
  áudio esperando processamento; `passo_ms`: duração da última volta). Uma
  estação que fica para trás age sobre um "agora" velho.
* **Comandos.** Um servidor TCP em 127.0.0.1 (porta padrão 7110; 7099 é a do
  aviso) aceita uma linha por conexão:

      estado            o JSON do estado
      log [n]           as últimas n linhas do diário (padrão 40)
      desde <k>         as linhas a partir da k-ésima (para acompanhar)
      pausa             para de tocar e congela a estação (ver abaixo)
      retoma            volta a alimentar a estação
      finaliza          sai assim que nada estiver tocando nem agendado
      para              sai já; grava tudo pelo mesmo caminho de sempre

  O servidor nunca toca na `Estacao`: só põe o comando numa fila que o laço
  principal lê entre uma volta e outra, e só lê um dicionário que o laço
  republica. Por isso um comando leva até ~0,5 s para valer, e **se o laço
  estiver preso dentro de `passo()` ele não vale nunca** -- `passo_ms` e
  `atraso_s` crescendo mostram isso, e aí o recurso é matar o processo
  (`taskkill /pid <pid> /f`; o pid está no estado) e rodar `recupera`.

**Pausa congela a estação e continua gravando.** O relógio da estação é o
número de amostras que ela ouviu, então enquanto ela não é alimentada o tempo
dela não anda: na volta, os prazos e as janelas cegas continuam de onde
estavam, e o que soou durante a pausa está no WAV mas a estação nunca o ouviu.
A outra ponta não pausa; espere que ela tenha seguido em frente.

Cliente, na mesma máquina:

    python painel.py estado
    python painel.py log 60
    python painel.py segue            # acompanha o diário até Ctrl-C
    python painel.py para
    python painel.py recupera <pasta>/<stem>.parcial.f32
"""

import json
import os
import queue
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

PORTA = 7110
COMANDOS = ('pausa', 'retoma', 'finaliza', 'para')


class Painel:
    """O lado da sessão: disco contínuo, estado publicado, fila de comandos.

    Só o laço principal chama `audio`, `linha` e `publica`; a thread do
    servidor só lê `_estado`/`_linhas` sob a trava e só escreve na fila.
    """

    def __init__(self, stem, porta=PORTA, fs=48000):
        self.stem = Path(stem)
        self.stem.parent.mkdir(parents=True, exist_ok=True)
        self.fs = fs
        self.t0 = time.time()
        self._f32 = open(self._nome('.parcial.f32'), 'wb')
        self._n = 0                    # amostras já em .parcial.f32
        # O que esta estação tocou, para o visor mostrar ao lado do que ouviu:
        # as amostras em .saida.f32 e, em .saida.idx, uma linha por trecho com
        # a amostra de .parcial.f32 em que ele começa a soar, onde ele está em
        # .saida.f32 e quantas amostras tem ("corte N": o que soaria depois de
        # N foi calado).
        self._sf32 = open(self._nome('.saida.f32'), 'wb')
        self._sidx = open(self._nome('.saida.idx'), 'w', encoding='utf-8')
        self._sn = 0
        self._txt = open(self._nome('.diario.txt'), 'w', encoding='utf-8')
        self._trava = threading.Lock()
        self._linhas = []
        self._estado = {}
        self._publicado = 0.0
        self.comandos = queue.Queue()
        self.porta = None
        if porta:
            self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._srv.bind(('127.0.0.1', porta))
            self._srv.listen(4)
            self.porta = porta
            threading.Thread(target=self._serve, daemon=True).start()

    def _nome(self, sufixo):
        return self.stem.with_name(self.stem.name + sufixo)

    # --- laço principal

    def audio(self, x):
        x = np.asarray(x, dtype=np.float32)
        self._f32.write(x.tobytes())
        self._n += len(x)

    def saida(self, x, atraso):
        """`x` começa a soar `atraso` amostras depois do que já foi ouvido."""
        x = np.asarray(x, dtype=np.float32)
        self._sf32.write(x.tobytes())
        self._sidx.write(f"{self._n + int(atraso)} {self._sn} {len(x)}\n")
        self._sn += len(x)

    def corta(self, atraso=0):
        self._sidx.write(f"corte {self._n + int(atraso)}\n")

    def linha(self, s):
        s = f"{time.time() - self.t0:8.2f} {s}"
        with self._trava:
            self._linhas.append(s)
        self._txt.write(s + "\n")
        self._txt.flush()
        return s

    def publica(self, estado, forca=False):
        agora = time.time()
        estado = dict(estado, pid=os.getpid(), porta=self.porta,
                      parede_s=round(agora - self.t0, 2),
                      atualizado=time.strftime("%H:%M:%S"))
        with self._trava:
            self._estado = estado
        if forca or agora - self._publicado >= 1.0:
            self._publicado = agora
            self._f32.flush()
            self._sf32.flush()
            self._sidx.flush()
            tmp = self._nome('.estado.json.tmp')
            # utf-8 explícito: no Windows o padrão é cp1252, e um "→" derrubaria a sessão.
            try:
                tmp.write_text(json.dumps(estado, indent=1, ensure_ascii=False),
                               encoding='utf-8')
            except OSError:
                return
            # No Windows o replace pode falhar com "Acesso negado" enquanto o visor lê
            # o arquivo. O estado é só vitrine: tenta de novo e, se não der, fica para a
            # próxima publicação.
            for _ in range(5):
                try:
                    os.replace(tmp, self._nome('.estado.json'))
                    break
                except PermissionError:
                    time.sleep(0.02)

    def fecha(self):
        """Depois do WAV definitivo gravado: o `.f32` vira redundante."""
        self._f32.close()
        self._txt.close()
        self._sf32.close()
        self._sidx.close()
        # O servidor também: no modo simulado o processo sobrevive à sessão
        # (a outra ponta continua), e um painel que ainda responde diz ao
        # visor que há uma sessão rodando.
        if self.porta:
            try:
                self._srv.close()
            except OSError:
                pass
        for suf in ('.parcial.f32', '.saida.f32', '.saida.idx'):
            try:
                self._nome(suf).unlink()
            except OSError:
                pass

    # --- servidor

    def _serve(self):
        while True:
            try:
                c, _ = self._srv.accept()
            except OSError:
                return              # fechado por `fecha`
            try:
                c.settimeout(5)
                linha = c.makefile('r', encoding='utf-8').readline().rstrip('\r\n')
                if linha.startswith('envia '):
                    # Texto digitado: vai inteiro, com os espaços.
                    self.comandos.put(('envia', linha[6:]))
                    c.sendall("ok: texto na fila\n".encode('utf-8'))
                    continue
                pedido = linha.split()
                c.sendall((self._responde(pedido) + "\n").encode('utf-8'))
            except Exception as e:
                try:
                    c.sendall(f"ERRO: {e}\n".encode())
                except OSError:
                    pass
            finally:
                c.close()

    def _responde(self, pedido):
        if not pedido:
            return "ERRO: comando vazio"
        cmd, resto = pedido[0], pedido[1:]
        with self._trava:
            if cmd == 'estado':
                return json.dumps(self._estado, indent=1, ensure_ascii=False)
            if cmd == 'log':
                n = int(resto[0]) if resto else 40
                return "\n".join(self._linhas[-n:])
            if cmd == 'desde':
                k = int(resto[0]) if resto else 0
                return json.dumps({'total': len(self._linhas), 'linhas': self._linhas[k:]},
                                  ensure_ascii=False)
        if cmd in COMANDOS:
            self.comandos.put(cmd)
            return f"ok: {cmd} na fila (vale na próxima volta do laço)"
        return f"ERRO: comando desconhecido {cmd!r}; conheço estado, log, desde, " + \
            ", ".join(COMANDOS)


# --- cliente

def pede(linha, porta=PORTA, timeout=10):
    with socket.create_connection(('127.0.0.1', porta), timeout=timeout) as c:
        c.sendall((linha + "\n").encode('utf-8'))
        partes = []
        while True:
            b = c.recv(65536)
            if not b:
                break
            partes.append(b)
    return b"".join(partes).decode('utf-8').rstrip("\n")


def segue(porta=PORTA):
    k = 0
    while True:
        try:
            r = json.loads(pede(f"desde {k}", porta))
        except (OSError, ValueError):
            print("[painel] sessão não responde (terminou?)", flush=True)
            return
        for s in r['linhas']:
            print(s, flush=True)
        k = r['total']
        time.sleep(1.0)


def recupera(f32, fs=48000):
    """`.parcial.f32` de uma sessão morta -> o WAV de `recording`."""
    import recording
    f32 = Path(f32)
    x = np.fromfile(f32, dtype=np.float32)
    stem = Path(str(f32)[:-len('.parcial.f32')])
    wav = stem.with_name(stem.name + '.wav')
    wav.write_bytes(recording._wav_bytes(x, fs))
    print(f"[painel] {len(x) / fs:.1f} s -> {wav}")


def main():
    a = sys.argv[1:]
    porta = PORTA
    if '--porta' in a:
        i = a.index('--porta')
        porta = int(a[i + 1])
        del a[i:i + 2]
    if not a:
        print(__doc__)
        return 2
    try:
        if a[0] == 'segue':
            segue(porta)
        elif a[0] == 'recupera':
            recupera(a[1])
        else:
            print(pede(" ".join(a), porta))
    except ConnectionRefusedError:
        print(f"[painel] nada escutando em 127.0.0.1:{porta} -- a sessão não está rodando")
        return 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
