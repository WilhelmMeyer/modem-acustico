"""On-screen notice while a program is using the microphone or the loudspeaker.

Puts a notice on screen for exactly the interval the audio device is in use,
and replaces it with a short "liberado" when it is released.

One backend on both systems: a small always-on-top banner in the corner,
drawn with tkinter by this same file run as a child process, there only while
the device is in use. It closes when told to, and also when its parent dies --
it watches its own stdin, a pipe from the parent -- so a crashed program cannot
leave it on screen.

When an `aviso.py --servidor` is listening on localhost (Windows, see
`_Remoto`), the notice goes there instead of a banner of our own.

Runtime only: no DSP, and nothing here may fail or slow the caller. `inicio`/
`fim` only post the wanted state to a worker thread, so they return at once.
Do not call them from a PortAudio callback all the same. With no display at
all every call is a no-op.

The notice is silent: a chime would land in the recording it announces.

    with aviso.em_uso("microfone + alto-falante", "selfcapture, teste 3/8"):
        ...
"""

import atexit
import contextlib
import os
import queue
import subprocess
import sys
import threading

_WIN = sys.platform == 'win32'
_DESKTOP = _WIN or bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
_ATIVO = _DESKTOP and os.environ.get('AUDIOFSK_AVISO', '1') != '0'

PORTA = 7099      # aviso.py --servidor, localhost only

_q = queue.Queue()
_worker = None
_lock = threading.Lock()


# --- backends: run only on the worker thread

class _Banner:
    def __init__(self):
        self.proc = None

    def _kill(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()      # an 'on' banner closes on EOF
                self.proc.wait(timeout=0.5)
            except Exception:                # noqa: BLE001
                self.proc.kill()
        self.proc = None

    def show(self, titulo, corpo, ativo):
        self._kill()
        flags = 0x08000000 if _WIN else 0    # CREATE_NO_WINDOW: no console flash
        exe = sys.executable
        if _WIN and exe.lower().endswith('python.exe'):
            alt = exe[:-len('python.exe')] + 'pythonw.exe'
            exe = alt if os.path.exists(alt) else exe
        self.proc = subprocess.Popen(
            [exe, os.path.abspath(__file__), '--banner',
             'on' if ativo else 'off', titulo, corpo],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, creationflags=flags)

    def close(self):
        self._kill()


class _Remoto:
    """Hand the notice to `aviso.py --servidor`, falling back to a banner of
    our own when nothing is listening.

    Needed on Windows for a process started over SSH: it lives in session 0,
    which has no desktop, so a window it opens is shown to nobody. The server
    runs in the logged-in session (a logon task, see `--instala`) and draws
    there. The connection stays open while the notice is 'on', so a client that
    dies takes its banner down with it.
    """

    def __init__(self):
        self.sock = None
        self.local = _Banner()

    def show(self, titulo, corpo, ativo):
        import socket
        linha = ('\t'.join(('on' if ativo else 'off', titulo, corpo)) + '\n').encode()
        for _ in range(2):                   # one reconnect if the server restarted
            try:
                if self.sock is None:
                    self.sock = socket.create_connection(('127.0.0.1', PORTA), timeout=0.5)
                self.sock.sendall(linha)
                self.local.close()
                return
            except OSError:
                self.close_sock()
        self.local.show(titulo, corpo, ativo)

    def close_sock(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def close(self):
        self.close_sock()
        self.local.close()


def _run():
    backend = _Remoto()
    while True:
        items = [_q.get()]
        while True:                          # coalesce: the latest state wins
            try:
                items.append(_q.get_nowait())
            except queue.Empty:
                break
        try:
            if None in items:
                backend.close()
            else:
                backend.show(*items[-1])
        except Exception:                    # noqa: BLE001 - never fail a test
            pass
        for _ in items:
            _q.task_done()
        if None in items:
            return


def _post(item):
    global _worker
    if not _ATIVO:
        return
    with _lock:
        if _worker is None:
            _worker = threading.Thread(target=_run, name='aviso', daemon=True)
            _worker.start()
            atexit.register(_flush)
    _q.put(item)


def _flush():
    """At exit, let the last notice reach the screen, then take a banner down
    once it has been read."""
    try:
        # The 'off' banner times itself out; just let it be spawned.
        import time
        t0 = time.time()
        while _q.unfinished_tasks and time.time() - t0 < 3:
            time.sleep(0.05)
    except Exception:                        # noqa: BLE001
        pass


def inicio(o_que, detalhe=''):
    _post((f"🔴 {o_que} EM USO", detalhe or "teste de áudio em andamento", True))


def fim(detalhe=''):
    _post(("✅ áudio liberado", detalhe or "teste terminou", False))


@contextlib.contextmanager
def em_uso(o_que, detalhe=''):
    inicio(o_que, detalhe)
    try:
        yield
    finally:
        fim(detalhe)


# --- the banner itself, when this file runs as a child process

def _desenha(root, titulo, corpo, ativo):
    import tkinter as tk
    bg = '#c62828' if ativo else '#2e7d32'
    for w in root.winfo_children():
        w.destroy()
    frame = tk.Frame(root, bg=bg, padx=16, pady=10)
    frame.pack()
    fonte = 'Segoe UI' if _WIN else 'Sans'
    tk.Label(frame, text=titulo, bg=bg, fg='white', font=(fonte, 14, 'bold')).pack(anchor='w')
    tk.Label(frame, text=corpo, bg=bg, fg='white', font=(fonte, 10)).pack(anchor='w')
    root.update_idletasks()
    w = root.winfo_reqwidth()
    root.geometry(f"+{root.winfo_screenwidth() - w - 24}+{24}")
    return frame


def _janela():
    import tkinter as tk
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes('-topmost', True)
    if _WIN:
        try:
            root.attributes('-toolwindow', True)
        except tk.TclError:
            pass
    return root


def _servidor():
    """Draw notices sent by other processes, in this (the desktop's) session."""
    import socket
    root = _janela()
    root.withdraw()
    eventos = queue.Queue()
    esconder = [None]

    def atende(conn):
        ultimo = None
        with conn:
            arq = conn.makefile('r', encoding='utf-8', errors='replace')
            for linha in arq:
                partes = linha.rstrip('\n').split('\t')
                if len(partes) == 3:
                    ultimo = partes
                    eventos.put(partes)
        if ultimo and ultimo[0] == 'on':      # client died mid-use
            eventos.put(['off', '⚠ aviso encerrado', ultimo[2] + ' (processo terminou)'])

    def escuta():
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', PORTA))
        srv.listen()
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=atende, args=(conn,), daemon=True).start()

    threading.Thread(target=escuta, daemon=True).start()

    def poll():
        try:
            while True:
                estado, titulo, corpo = eventos.get_nowait()
                ativo = estado == 'on'
                if esconder[0] is not None:
                    root.after_cancel(esconder[0])
                    esconder[0] = None
                frame = _desenha(root, titulo, corpo, ativo)
                frame.bind('<Button-1>', lambda e: root.withdraw())
                root.deiconify()
                root.attributes('-topmost', True)
                if not ativo:
                    esconder[0] = root.after(3000, root.withdraw)
        except queue.Empty:
            pass
        root.after(100, poll)

    root.after(100, poll)
    root.mainloop()


def _instala():
    """Windows: register the server as a logon task in the desktop session,
    and start it now."""
    exe = sys.executable
    if exe.lower().endswith('python.exe'):
        alt = exe[:-len('python.exe')] + 'pythonw.exe'
        exe = alt if os.path.exists(alt) else exe
    tr = f'"{exe}" "{os.path.abspath(__file__)}" --servidor'
    subprocess.run(['schtasks', '/create', '/f', '/tn', 'audioFSK-aviso',
                    '/sc', 'onlogon', '/it', '/tr', tr], check=True)
    subprocess.run(['schtasks', '/run', '/tn', 'audioFSK-aviso'], check=True)


def _banner(estado, titulo, corpo):
    ativo = estado == 'on'
    root = _janela()
    frame = _desenha(root, titulo, corpo, ativo)

    # Parent gone or told us to go: stdin reaches EOF.
    done = threading.Event()

    def watch():
        try:
            sys.stdin.buffer.read()
        except Exception:                    # noqa: BLE001
            pass
        done.set()

    # Only the 'on' banner follows its parent; 'liberado' outlives it so the
    # last notice of a run is still read after the process has exited.
    if ativo and sys.stdin is not None:
        threading.Thread(target=watch, daemon=True).start()

    def poll():
        if done.is_set():
            root.destroy()
            return
        root.after(100, poll)

    root.after(100, poll)
    if not ativo:
        root.after(3000, root.destroy)       # "liberado" reads, then goes
    frame.bind('<Button-1>', lambda e: root.destroy())
    root.mainloop()


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--banner':
        _banner(*sys.argv[2:])
    elif sys.argv[1:] == ['--servidor']:
        _servidor()
    elif sys.argv[1:] == ['--instala']:
        _instala()
    else:
        sys.exit("uso: aviso.py --servidor | --instala")
