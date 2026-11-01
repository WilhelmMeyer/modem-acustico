"""Visor do enlace: uma tela no navegador para uma sessão de `saudacao.py`.

Roda em cada máquina, ao lado da sessão daquela máquina, e só escuta em
127.0.0.1: cada tela se abre na própria máquina.

    ./venv/bin/python visor.py                 # abre http://127.0.0.1:7120
    venv\\Scripts\\python.exe visor.py --dir <pasta>

O que a tela mostra vem de três lugares, e de nenhum outro:

* o **painel** da sessão (`painel.py`, 127.0.0.1:7110): `estado` e `desde`;
* o **disco** da sessão: `<stem>.estado.json` e `<stem>.diario.txt` quando o
  painel já não responde (a sessão terminou), e o `<stem>.parcial.f32`, que é
  o microfone cru, para a forma de onda e o espectrograma;
* o arquivo `--receber-em` que esta tela passa ao lançar a sessão, para mostrar
  o que chegou -- a sessão só o escreve quando conclui.

Não importa `enlace` nem toca na `Estacao`: lança `saudacao.py` como processo
filho, com os mesmos argumentos que se daria à mão, e manda os comandos do
painel (`pausa`, `retoma`, `finaliza`, `para`) pelo próprio painel. Uma sessão
lançada de outro jeito aparece igual, só sem o botão de iniciar disponível
enquanto ela roda.
"""

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

import painel

AQUI = Path(__file__).resolve().parent
PAGINA = AQUI / 'visor'
FS = 48000
PORTA = 7120
PAINEL = painel.PORTA     # a mesma de --painel-porta da sessão; 0 = nenhuma
SIMULADO = None           # --simulado: o iniciar lança as duas pontas sem som

# Espectrograma: as mesmas contas para toda coluna, para que duas colunas
# sejam comparáveis. 4096 amostras = 11,7 Hz por raia, 85 ms de janela; passo
# de 2048 = 42,7 ms por coluna.
NFFT = 4096
PASSO = 2048
F_MIN, F_MAX = 400.0, 4600.0
N_RAIAS = 280
_JANELA = np.hanning(NFFT).astype(np.float32)
_FREQS = np.fft.rfftfreq(NFFT, 1 / FS)
_ALVO = np.linspace(F_MIN, F_MAX, N_RAIAS)
# Seno de amplitude 1 dá |X| = soma da janela / 2: é o 0 dBFS de um tom.
_REF = float(_JANELA.sum() / 2)

MODOS_DADOS = ('auto', '16fsk', '2fsk', 'votada', 'multicanal')


class Sessao:
    """O processo filho que esta tela lançou, se lançou algum."""

    def __init__(self, base, saida):
        self.base = Path(base)
        self.saida = self.base / saida
        self.proc = None
        self.args = None
        self.recebido = None
        self.iniciada = None
        self.trava = threading.Lock()

    def rodando(self):
        return self.proc is not None and self.proc.poll() is None

    def inicia(self, p):
        with self.trava:
            if self.rodando():
                raise RuntimeError("já há uma sessão lançada por esta tela")
            if painel_vivo() or (SIMULADO and painel_vivo(PAINEL + 1)):
                raise RuntimeError("já há uma sessão nesta máquina (o painel responde em "
                                   f"127.0.0.1:{PAINEL}); pare-a antes")
            if not SIMULADO:
                # Esta pasta é de sessões simuladas (a outra ponta de um
                # `--simulado`): o Iniciar daqui abriria o microfone e a caixa
                # de verdade para quem só queria ver a simulação.
                ultimo = stem_recente(self.saida)
                try:
                    sim = ultimo and json.loads(Path(str(ultimo) + '.estado.json').read_text(
                        encoding='utf-8')).get('simulado')
                except (OSError, ValueError):
                    sim = None
                if sim:
                    raise RuntimeError(f"as sessões desta pasta são simuladas ({sim}); "
                                       "lance o visor com --simulado para iniciar outra")
            papel = p.get('papel')
            if papel not in ('chamador', 'ouvinte'):
                raise ValueError("papel tem de ser chamador ou ouvinte")
            modo = p.get('modo_dados') or 'auto'
            if modo not in MODOS_DADOS:
                raise ValueError(f"modo de dados desconhecido: {modo}")
            self.saida.mkdir(parents=True, exist_ok=True)
            marca = time.strftime("%Y%m%d-%H%M%S")
            self.recebido = self.saida / f"{marca}-visor-recebido.bin"
            a = [sys.executable, '-u', str(self.base / 'saudacao.py'),
                 '--papel', papel, '--modo-dados', modo, '--painel-porta', str(PAINEL),
                 '--out', str(self.saida), '--receber-em', str(self.recebido),
                 '--log', str(self.saida / 'visor-saudacao.log')]
            texto = p.get('texto') or ''
            if texto:
                a += ['--enviar', texto]
            if p.get('arquivo_b64'):
                nome = Path(p.get('arquivo_nome') or 'arquivo.bin').name
                dest = self.saida / f"{marca}-visor-envia-{nome}"
                dest.write_bytes(base64.b64decode(p['arquivo_b64']))
                a += ['--enviar-arquivo', str(dest)]
            if p.get('conversa'):
                a += ['--conversa']
            for chave, opc in (('duracao', '--duracao'), ('fim_silencio', '--fim-silencio')):
                v = p.get(chave)
                if v not in (None, ''):
                    a += [opc, str(float(v))]
            if p.get('manual'):
                a += ['--manual']
                for chave, opc in (('modulacao', '--modulacao'), ('baud', '--baud'), ('rep', '--rep'),
                                   ('tom_base', '--tom-base'), ('passo_tom', '--passo-tom'),
                                   ('caixa', '--caixa')):
                    if p.get(chave) not in (None, ''):
                        a += [opc, str(p[chave])]
                if p.get('sem_confirmacao'):
                    a += ['--sem-confirmacao']
            if SIMULADO:
                # Sem microfone nem alto-falante: a outra ponta roda no mesmo
                # processo, com o painel na porta seguinte e a pasta ao lado.
                a += ['--simulado', SIMULADO, '--parceiro-out', str(self.saida) + '-parceiro']
            else:
                for chave, opc in (('in_device', '--in-device'), ('out_device', '--out-device')):
                    if p.get(chave):
                        a += [opc, p[chave]]
            env = dict(os.environ)
            env['PYTHONPATH'] = os.pathsep.join(
                x for x in (str(AQUI), env.get('PYTHONPATH', '')) if x)
            # O diário detalhado (ordem das marcas, fim de turno) é o que
            # explica uma sessão no ar depois; sem ele, só se vê o resumo.
            env.setdefault('ENLACE_DEPURA', '1')
            self.proc = subprocess.Popen(a, cwd=str(self.base), env=env,
                                         stdin=subprocess.DEVNULL,
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
            self.args = a[2:]
            self.iniciada = time.time()
            return self.proc.pid

    def mata(self):
        if self.rodando():
            self.proc.kill()

    def resumo(self):
        if self.proc is None:
            return None
        return dict(pid=self.proc.pid, rodando=self.rodando(),
                    codigo=self.proc.poll(), args=self.args,
                    ha_s=round(time.time() - self.iniciada, 1))


def painel_vivo(porta=None):
    porta = PAINEL if porta is None else porta
    if not porta:
        return False
    try:
        painel.pede('estado', porta, timeout=0.6)
        return True
    except OSError:
        return False


def pede_json(linha):
    if not PAINEL:
        return None
    try:
        r = painel.pede(linha, PAINEL, timeout=1.5)
    except OSError:
        return None
    try:
        return json.loads(r)
    except ValueError:
        return None


def stem_recente(pasta):
    """A sessão mais recente da pasta, pelo `.estado.json` que o painel reescreve."""
    try:
        c = sorted(Path(pasta).glob('*.estado.json'), key=lambda f: f.stat().st_mtime)
    except OSError:
        return None
    if not c:
        return None
    return Path(str(c[-1])[:-len('.estado.json')])


def tocado(stem, ini, n):
    """O que a sessão tocou entre as amostras `ini` e `n` da entrada, na
    linha do tempo de `.parcial.f32` (zeros onde nada soava); None se a
    sessão não grava a saída ou nada soou no intervalo."""
    idx = Path(str(stem) + '.saida.idx')
    f32 = Path(str(stem) + '.saida.f32')
    try:
        linhas = idx.read_text(encoding='utf-8').splitlines()
    except OSError:
        return None
    y = np.zeros(max(0, n - ini), dtype=np.float32)
    trechos, achou = [], False
    for l in linhas:
        c = l.split()
        if len(c) == 2 and c[0] == 'corte':
            # o que ainda soaria depois do corte foi calado
            corte = int(c[1])
            trechos = [(a, o, min(m, max(0, corte - a))) for a, o, m in trechos]
        elif len(c) == 3:
            trechos.append(tuple(int(v) for v in c))
    try:
        fh = open(f32, 'rb')
    except OSError:
        return None
    with fh:
        for a, o, m in trechos:
            lo, hi = max(a, ini), min(a + m, n)
            if hi <= lo:
                continue
            fh.seek((o + lo - a) * 4)
            seg = np.frombuffer(fh.read((hi - lo) * 4), dtype=np.float32)
            y[lo - ini:lo - ini + len(seg)] += seg
            achou = True
    return y if achou else None


def espectro(x):
    """dB de um tom de amplitude 1 = 0, nas N_RAIAS frequências da escala."""
    X = np.abs(np.fft.rfft(x[-NFFT:] * _JANELA)) / _REF
    db = 20 * np.log10(np.interp(_ALVO, _FREQS, X) + 1e-9)
    return db


class Visor:
    def __init__(self, base, saida):
        self.sessao = Sessao(base, saida)

    def stem(self):
        return stem_recente(self.sessao.saida)

    # --- respostas da API

    def estado(self):
        e = pede_json('estado')
        vivo = e is not None
        stem = self.stem()
        if e is None and stem is not None:
            try:
                e = json.loads(Path(str(stem) + '.estado.json').read_text(encoding='utf-8'))
            except (OSError, ValueError):
                e = None
        sim = SIMULADO or (e or {}).get('simulado')
        maquina = socket.gethostname() + (f" · SIMULADA ({sim})" if sim else "")
        return dict(vivo=vivo, estado=e, stem=stem.name if stem else None,
                    sessao=self.sessao.resumo(), maquina=maquina, simulado=sim,
                    agora=time.strftime("%H:%M:%S"))

    def diario(self, desde):
        r = pede_json(f'desde {desde}')
        if r is not None:
            return dict(r, fonte='painel')
        stem = self.stem()
        if stem is None:
            return dict(total=0, linhas=[], fonte='nada')
        try:
            linhas = Path(str(stem) + '.diario.txt').read_text(
                encoding='utf-8').splitlines()
        except OSError:
            linhas = []
        return dict(total=len(linhas), linhas=linhas[desde:], fonte='disco')

    def audio(self, desde, stem_pedido, cabe=40):
        """Colunas do espectrograma a partir da amostra `desde`, mais a forma
        de onda dos últimos 200 ms. `desde` < 0 quer dizer: só o fim."""
        stem = self.stem()
        f32 = Path(str(stem) + '.parcial.f32') if stem else None
        if f32 is None or not f32.exists():
            return dict(stem=stem.name if stem else None, ativo=False)
        n = f32.stat().st_size // 4
        # Na primeira vez, o que cabe no papel da tela; depois, no máximo isso
        # também -- uma aba esquecida que volta não pede minutos de áudio.
        cabe = max(1, min(int(cabe), 600))
        if stem.name != stem_pedido or desde < 0 or desde > n:
            desde = n - NFFT - cabe * PASSO
        desde = max(0, desde, n - NFFT - cabe * PASSO)
        ini = max(0, desde)
        with open(f32, 'rb') as fh:
            fh.seek(ini * 4)
            x = np.frombuffer(fh.read((n - ini) * 4), dtype=np.float32)
        y = tocado(stem, ini, n)
        cols, envio = [], []
        k = 0
        while k + NFFT <= len(x):
            cols.append(np.clip(np.round(espectro(x[k:k + NFFT])), -140, 0).astype(int).tolist())
            # o que esta estação tocava na mesma janela; None em silêncio
            yk = y[k:k + NFFT] if y is not None else None
            envio.append(np.clip(np.round(espectro(yk)), -140, 0).astype(int).tolist()
                         if yk is not None and np.any(yk) else None)
            k += PASSO
        prox = ini + k
        cauda = x[-int(0.2 * FS):] if len(x) else np.zeros(1, np.float32)
        if len(cauda) < int(0.2 * FS):
            with open(f32, 'rb') as fh:
                fh.seek(max(0, n - int(0.2 * FS)) * 4)
                cauda = np.frombuffer(fh.read(), dtype=np.float32)
        m = 360
        if len(cauda) >= m:
            b = cauda[:len(cauda) // m * m].reshape(m, -1)
            forma = np.stack([b.min(1), b.max(1)], 1).round(4).tolist()
        else:
            forma = []
        rms = float(np.sqrt(np.mean(cauda ** 2))) if len(cauda) else 0.0
        pico = float(np.max(np.abs(cauda))) if len(cauda) else 0.0
        agora = espectro(cauda) if len(cauda) >= NFFT else None
        dom = None
        if agora is not None:
            i = int(np.argmax(agora))
            # Só aponta se o pico se destaca do chão do próprio espectro.
            if agora[i] - np.median(agora) > 18:
                dom = round(float(_ALVO[i]), 1)
        return dict(stem=stem.name, ativo=True, prox=prox, n=n,
                    colunas=cols, envio=envio, forma=forma, rms=round(rms, 5), pico=round(pico, 4),
                    dominante=dom, f_min=F_MIN, f_max=F_MAX, raias=N_RAIAS,
                    passo_s=PASSO / FS)

    def recebido(self):
        f = self.sessao.recebido
        if f is None or not f.exists():
            return dict(existe=False)
        b = f.read_bytes()
        try:
            texto = b.decode('utf-8')
            legivel = True
        except UnicodeDecodeError:
            texto = b.decode('latin-1')
            legivel = False
        r = dict(existe=True, nome=f.name, bytes=len(b), legivel=legivel,
                 texto=texto[:4000], hex=b[:512].hex())
        if b[:2] == b'BM' and len(b) >= 6:
            # Um BMP diz o próprio tamanho: com o resto em zeros já abre, e a
            # imagem sobe linha a linha enquanto os quadros chegam.
            total = int.from_bytes(b[2:6], 'little')
            r['total'] = total
            r['bmp'] = base64.b64encode(b.ljust(total, b'\0')).decode('ascii')
        return r

    def dispositivos(self):
        try:
            import sounddevice as sd
            d = sd.query_devices()
        except Exception as e:
            return dict(erro=str(e), entrada=[], saida=[])
        ent = sorted({x['name'] for x in d if x['max_input_channels'] > 0})
        sai = sorted({x['name'] for x in d if x['max_output_channels'] > 0})
        return dict(entrada=ent, saida=sai)

    def comando(self, cmd):
        if cmd == 'mata':
            self.sessao.mata()
            return "processo da sessão terminado à força"
        if cmd not in painel.COMANDOS:
            raise ValueError(f"comando desconhecido: {cmd}")
        try:
            return painel.pede(cmd, PAINEL, timeout=3)
        except OSError:
            raise RuntimeError("o painel não responde: não há sessão rodando")


def faz_handler(v):
    tipos = {'.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
             '.js': 'text/javascript; charset=utf-8', '.woff2': 'font/woff2',
             '.svg': 'image/svg+xml', '.txt': 'text/plain; charset=utf-8'}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _manda(self, corpo, tipo='application/json; charset=utf-8', codigo=200):
            if not isinstance(corpo, bytes):
                corpo = json.dumps(corpo, ensure_ascii=False).encode('utf-8')
            self.send_response(codigo)
            self.send_header('Content-Type', tipo)
            self.send_header('Content-Length', str(len(corpo)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(corpo)

        def _erro(self, e, codigo=400):
            self._manda(dict(erro=str(e)), codigo=codigo)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == '/api/estado':
                    return self._manda(v.estado())
                if u.path == '/api/diario':
                    return self._manda(v.diario(int(q.get('desde', ['0'])[0])))
                if u.path == '/api/audio':
                    return self._manda(v.audio(int(q.get('desde', ['-1'])[0]),
                                               q.get('stem', [''])[0],
                                               int(q.get('cabe', ['40'])[0])))
                if u.path == '/api/recebido':
                    return self._manda(v.recebido())
                if u.path == '/api/dispositivos':
                    return self._manda(v.dispositivos())
            except Exception as e:
                return self._erro(e, 500)
            nome = 'index.html' if u.path in ('/', '') else u.path.lstrip('/')
            f = (PAGINA / nome).resolve()
            if PAGINA.resolve() not in f.parents or not f.is_file():
                return self._manda(b'nada aqui', 'text/plain', 404)
            self._manda(f.read_bytes(), tipos.get(f.suffix, 'application/octet-stream'))

        def do_POST(self):
            u = urlparse(self.path)
            try:
                n = int(self.headers.get('Content-Length') or 0)
                p = json.loads(self.rfile.read(n) or b'{}')
                if u.path == '/api/inicia':
                    return self._manda(dict(pid=v.sessao.inicia(p)))
                if u.path == '/api/comando':
                    return self._manda(dict(resposta=v.comando(p.get('cmd', ''))))
                if u.path == '/api/envia':
                    t = ' '.join(str(p.get('texto', '')).splitlines()).strip()
                    if not t:
                        raise ValueError("texto vazio")
                    try:
                        return self._manda(dict(resposta=painel.pede('envia ' + t, PAINEL, timeout=3)))
                    except OSError:
                        raise RuntimeError("o painel não responde: não há sessão rodando")
            except (ValueError, RuntimeError) as e:
                return self._erro(e)
            except Exception as e:
                return self._erro(e, 500)
            self._erro('rota desconhecida', 404)

    return H


def main():
    global PAINEL, SIMULADO
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--porta', type=int, default=PORTA)
    ap.add_argument('--dir', default=str(AQUI),
                    help="onde está o saudacao.py que a tela lança e de onde ela roda "
                         "(em B: a pasta enlace_teste)")
    ap.add_argument('--out', default='sessoes-enlace',
                    help="pasta das sessões, relativa a --dir (a mesma do --out da sessão)")
    ap.add_argument('--painel-porta', type=int, default=PAINEL,
                    help="porta do painel das sessões (a --painel-porta de saudacao.py)")
    ap.add_argument('--sem-navegador', action='store_true')
    ap.add_argument('--simulado', default=None, metavar='CENARIO',
                    help="o Iniciar lança saudacao.py --simulado CENARIO (as duas pontas, "
                         "sem som); a outra ponta se vê com --painel-porta P+1 --out <out>-parceiro")
    args = ap.parse_args()
    PAINEL = args.painel_porta
    SIMULADO = args.simulado
    v = Visor(args.dir, args.out)
    srv = ThreadingHTTPServer(('127.0.0.1', args.porta), faz_handler(v))
    url = f"http://127.0.0.1:{args.porta}/"
    print(f"[visor] {url}  (sessões em {v.sessao.saida})", flush=True)
    if not args.sem_navegador:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
