"""A varredura de tons do `enlace2`, sem som: duas estações por dois `Canal`.

Mede o que a seleção compra num canal com tons mortos, contra a mesma sessão com
a tabela fixa, e o que ela custa num canal limpo. Imprime SUCCESS!/FAILED.

    ./venv/bin/python simula_tons.py
"""

import sys
import time

import numpy as np

import enlace2 as E
from simula_enlace import Canal

FS = E.FS
BLOCO = 2048
TEXTO_A = bytes(range(200)) * 2       # 400 B do chamador
TEXTO_B = b'resposta do ouvinte'


class Corte:
    """Um Canal com cortes estreitos, um por tom morto."""

    def __init__(self, rng, mortos):
        from scipy.signal import butter
        self.c = Canal(rng, perda=0.5, snr_ruido=0.003, ecos=2, limitador=0.6)
        self.sos = [butter(4, (f - 60, f + 60), btype='bandstop', fs=FS, output='sos') for f in mortos]

    def passa(self, x):
        from scipy.signal import sosfilt
        y = self.c.passa(x)
        for s in self.sos:
            y = sosfilt(s, y)
            y = sosfilt(s, y)
        return y


def roda(mortos_ab, mortos_ba, limite, fixa=False, surda=False, semente=1):
    rng = np.random.default_rng(semente)
    diario = []
    A = E.Estacao('chamador', semente=1, log=diario.append)
    B = E.Estacao('ouvinte', semente=2)
    A.enviar(TEXTO_A)
    B.enviar(TEXTO_B)
    ab, ba = Corte(rng, mortos_ab), Corte(rng, mortos_ba)
    entra = {id(A): np.zeros(0), id(B): np.zeros(0)}
    t = 0
    tons_t = {}
    original = E.escolhe_mapa
    if fixa:
        E.escolhe_mapa = lambda m: E.mapa_de(E.PADRAO)
    if surda:
        E.escolhe_mapa = lambda m: 0
    try:
        while t < limite * FS and not (A.encerrada and B.encerrada):
            for eu, outro, canal in ((A, B, ab), (B, A, ba)):
                f = entra[id(eu)]
                x = np.zeros(BLOCO)
                k = min(BLOCO, len(f))
                x[:k] = f[:k]
                entra[id(eu)] = f[k:]
                eu.ouvir((x + rng.normal(0, 0.002, BLOCO)).astype(np.float32))
                b = eu.passo()
                if b is not None:
                    if str(getattr(b, 'rotulo', '')) == 'VARREDURA':
                        tons_t.setdefault(id(eu), t)
                    y = canal.passa(E.enlace.aplica_caixa(b, E.enlace.CAIXA_SIM))
                    g = entra[id(outro)]
                    if len(g) < len(y):
                        g = np.concatenate([g, np.zeros(len(y) - len(g))])
                    g[:len(y)] += y
                    entra[id(outro)] = g
            t += BLOCO
    finally:
        E.escolhe_mapa = original
    return dict(t=t / FS, diario=diario, A=A, B=B, a_rec=len(A.recebido), b_rec=len(B.recebido), fim=A.encerrada and B.encerrada)


def fora(mapa):
    return [E.GRADE[k] for k in E.PADRAO if k not in E.tons_de(mapa or 0)] if mapa else 'não medido'


def resumo(nome, r):
    A = r['A']
    print(f"{nome}: {r['t']:.0f} s, {'fim' if r['fim'] else 'não terminou'}, "
          f"bytes {r['b_rec']}/{len(TEXTO_A)} e {r['a_rec']}/{len(TEXTO_B)}; "
          f"A->B {A.contrato_tx.descreve() if A.contrato_tx else '-'}, "
          f"B->A {A.contrato_rx.descreve() if A.contrato_rx else '-'}; "
          f"tons {duracao_tons(r['diario']):.0f} s; fora A->B {fora(A.reg['tons_mando'])}, B->A {fora(A.reg['tons_chega'])}")


def duracao_tons(diario):
    """Da entrada na etapa até a proposta da seguinte, pelo relógio da estação."""
    t = lambda frag: next(float(l.split(']')[0].split()[-1][:-1]) for l in diario if frag in l)
    return t('proponho etapa 4') - t('-> tons_ida')


def main():
    t0 = time.time()
    # Três tons da tabela padrão mortos em cada sentido.
    ab, ba = (1050.0, 1700.0, 2512.5), (1375.0, 2025.0, 2837.5)
    limite = 1500
    com = roda(ab, ba, limite)
    resumo('com seleção, tons mortos', com)
    sem = roda(ab, ba, limite, fixa=True)
    resumo('tabela fixa, tons mortos', sem)
    limpo = roda((), (), limite)
    resumo('com seleção, limpo', limpo)
    surdo = roda(ab, ba, limite, surda=True)
    resumo('medida ilegível', surdo)
    print(f"[{time.time() - t0:.0f} s de CPU]")
    ok = com['fim'] and com['b_rec'] == len(TEXTO_A) and com['a_rec'] == len(TEXTO_B)
    exc = set(E.PADRAO) - set(E.tons_de(com['A'].reg['tons_mando'] or 0))
    ok = ok and {E.GRADE.index(f) for f in ab} <= exc
    ok = ok and (not sem['fim'] or com['t'] < sem['t'])
    ok = ok and limpo['fim'] and limpo['A'].tons_tx is None and limpo['A'].tons_rx is None
    ok = ok and surdo['A'].reg['tons_mando'] == 0 and surdo['A'].tons_tx is None and surdo['fim']
    print("SUCCESS!" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
