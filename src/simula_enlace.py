"""Duas estações de `enlace.py` conversando por dois canais simulados.

Não é um teste do DSP, e sim do protocolo: a saudação, a sondagem, o contrato,
o canal aberto e as voltas atrás. Por isso o canal é adversário de propósito:

- dois sentidos diferentes (A->B pior que B->A);
- resposta em pente (ecos curtos) e uma cauda abaixo do tom;
- um limitador na saída, que faz o ganho alto perder para o baixo;
- relógios diferentes nas duas pontas (1e-4) e atraso de propagação aleatório;
- opcionalmente uma faixa morta, que derruba uma abordagem de controle e
  obriga o rodízio e as sombras;
- o próprio alto-falante entrando alto no próprio microfone, para provar que
  a cegueira durante a transmissão funciona;
- e um cenário sem parceiro, que nunca pode dizer "achei".

Imprime SUCCESS!/FAILED. Sem hardware.

    ./venv/bin/python simula_enlace.py            # todos os cenários
    ./venv/bin/python simula_enlace.py limpo -v   # um, com o diário das estações
    ./venv/bin/python simula_enlace.py limpo --semente=3   # outro sorteio de canal
"""

import sys
import time

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

import enlace as E

FS = E.FS
BLOCO = 1024


class Canal:
    """Um sentido do ar: saída limitada, pente, cauda, atraso, relógio."""

    def __init__(self, rng, perda=0.3, snr_ruido=0.003, ecos=3, cauda_db=-14.0,
                 relogio=0.0, morta=None, limitador=0.35, cauda_s=0.06):
        self.rng = rng
        self.perda = perda
        self.ruido = snr_ruido
        self.relogio = relogio
        self.limitador = limitador
        self.atraso = int(rng.uniform(0.0, 0.01) * FS)
        # Ecos de 1 a 8 ms: um pente com covas fundas na banda.
        h = np.zeros(int(max(0.12, 0.01 + 5 * cauda_s) * FS))
        h[0] = 1.0
        for _ in range(ecos):
            h[int(rng.uniform(0.001, 0.008) * FS)] += rng.uniform(-0.7, 0.7)
        # A cauda: ruído decaindo, `cauda_db` abaixo do direto, constante `cauda_s`.
        n = np.arange(len(h) - int(0.01 * FS))
        tail = rng.normal(0, 1, len(n)) * np.exp(-n / (cauda_s * FS))
        tail *= 10 ** (cauda_db / 20) / np.sqrt(np.sum(tail ** 2))
        h[int(0.01 * FS):] += tail
        self.h = h / np.sqrt(np.sum(h ** 2))
        self.sos_morta = (butter(6, morta, btype='bandstop', fs=FS, output='sos')
                          if morta else None)

    def passa(self, x):
        # Limitador do alto-falante: tanh, linear para baixo e achatando o pico.
        y = self.limitador * np.tanh(np.asarray(x) / self.limitador)
        y = fftconvolve(y, self.h)
        if self.sos_morta is not None:
            y = sosfilt(self.sos_morta, y)
        if self.relogio:
            n = len(y)
            t = np.arange(int(n * (1 + self.relogio))) / (1 + self.relogio)
            y = np.interp(t, np.arange(n), y)
        return np.concatenate([np.zeros(self.atraso), y * self.perda])


class Ar:
    """O que chega ao microfone de uma estação, indexado pelo tempo dela."""

    def __init__(self, rng, ruido, piora=None, rajadas=None, semente=0):
        self.x = np.zeros(0)
        self.rng = rng
        self.ruido = ruido
        self.piora = piora          # (instante em s, desvio do ruído, tons interferentes)
        # (por segundo, duração em s, desvio): estouros de ruído largo em
        # instantes ao acaso -- uma porta, uma cadeira, alguém falando. Um
        # sorteio à parte, para não mudar o canal dos outros cenários.
        self.rajadas = rajadas
        self._raj = []
        self._raj_rng = np.random.default_rng(semente + 200)

    def soma(self, inicio, y):
        fim = inicio + len(y)
        if fim > len(self.x):
            self.x = np.concatenate([self.x, np.zeros(fim - len(self.x))])
        self.x[inicio:fim] += y

    def le(self, a, b):
        r, tons = self.ruido, ()
        if self.piora and a >= self.piora[0] * FS:
            r, tons = self.piora[1], self.piora[2]
        out = self.rng.normal(0, r, b - a)
        n = np.arange(a, b) / FS
        for f, amp in tons:
            out += amp * np.sin(2 * np.pi * f * n)
        if self.rajadas:
            taxa, dur, desvio = self.rajadas
            while not self._raj or self._raj[-1] < b:
                t0 = (self._raj[-1] if self._raj else 0) + self._raj_rng.exponential(1 / taxa) * FS
                self._raj.append(int(t0))
            n = int(dur * FS)
            for t0 in self._raj:
                i, j = max(a, t0), min(b, t0 + n)
                if j > i:
                    out[i - a:j - a] += self._raj_rng.normal(0, desvio, j - i)
        k = min(b, len(self.x))
        if k > a:
            out[:k - a] += self.x[a:k]
        return out


def roda(cenario, verbose=False, semente=0):
    rng = np.random.default_rng(semente)
    log = print if verbose else None
    c = CENARIOS[cenario]
    modo = c.get('modo_dados', 'auto')
    a = E.Estacao('chamador', semente=semente + 1, log=log, nome='A', modo_dados=modo)
    b = None if c.get('sem_parceiro') else E.Estacao('ouvinte', semente=semente + 2,
                                                     log=log, nome='B', modo_dados=modo)
    ar_a = Ar(rng, c['ruido_a'], rajadas=c.get('rajadas'), semente=semente)
    ar_b = Ar(rng, c['ruido_b'], c.get('piora_b'), rajadas=c.get('rajadas'), semente=semente + 1)
    ab = Canal(rng, **c['ab'])
    ba = Canal(rng, **c['ba'])
    msg_a = bytes(rng.integers(0, 256, c.get('dados_a', 0)).astype(np.uint8))
    msg_b = bytes(rng.integers(0, 256, c.get('dados_b', 0)).astype(np.uint8))
    a.enviar(msg_a)
    if b:
        b.enviar(msg_b)
    t, fim = 0, int(c['segundos'] * FS)
    rng_perda = np.random.default_rng(semente + 100)   # à parte: não muda o canal
    rng_falsa = np.random.default_rng(semente + 300)   # idem
    perdidos = 0
    falas = {'A': [], 'B': []}       # (início, fim) do que cada uma tocou
    completo = None                  # instante em que tudo foi entregue
    apos_fim = {'A': 0, 'B': 0}      # turnos tocados depois de encerrada
    injetou = False
    t0 = time.time()
    while t < fim:
        for est, ar_meu, ar_outro, canal in ((a, ar_a, ar_b, ab), (b, ar_b, ar_a, ba)):
            if est is None:
                continue
            bloco = ar_meu.le(t, t + BLOCO)
            if est is b and rng_perda.random() < c.get('perde_b', 0.0):
                # O bloco que a entrada de B jogou fora (transbordo do
                # PortAudio): a estação nunca o ouve, e o relógio dela -- as
                # amostras ouvidas -- fica para trás do ar.
                perdidos += 1
                if c.get('compensa', True):
                    # O que `saudacao.py` faz ao ver o buraco (pelo instante
                    # do ADC de cada bloco): silêncio do tamanho perdido, no
                    # lugar dele, e o relógio da estação continua no do ar.
                    est.ouvir(np.zeros(BLOCO))
            else:
                est.ouvir(bloco)
            ja_encerrada = getattr(est, 'encerrada', False)
            x = est.passo()
            if x is not None:
                # O volume da caixa que a estação pediu, como fator: o
                # simulador não tem fader.
                x = E.aplica_caixa(x, E.CAIXA_SIM)
            if x is not None and ja_encerrada:
                # Depois de se despedir, uma estação só toca o que já estava
                # na fila (a resposta do ouvinte ao FIM), e uma vez.
                apos_fim[est.nome] += 1
            if x is not None:
                # Latência da saída que a estação não sabe (um alto-falante
                # Bluetooth declara menos do que tem): o som sai depois do
                # instante em que ela acha que tocou, para os dois ouvidos.
                oculta = c.get('latencia_oculta_b' if est is b else 'latencia_oculta',
                               c.get('latencia_oculta', 0.0))
                muda = c.get('latencia_b_depois')
                if est is b and muda and t >= muda[0] * FS:
                    # A latência de um alto-falante Bluetooth pode mudar depois
                    # da saudação.
                    oculta = muda[1]
                lat = t + BLOCO + int(oculta * FS)
                falas['A' if est is a else 'B'].append((lat, lat + len(x)))
                ar_outro.soma(lat, canal.passa(x))
                if (c.get('perde_contrato') and est is a and a.aberto and not a.confirmado
                        and len(a.ultimo_turno[0]) == a.ctrl.amostras):
                    # O turno do contrato de A com o CONTRATO na abordagem de
                    # controle: esse quadro (o primeiro segmento) chega a B
                    # debaixo de um estouro, toda vez -- a abordagem que
                    # passou na saudação deixou de passar neste sentido.
                    ar_outro.soma(lat, rng.normal(0, 2.0, a.ctrl.amostras + E.FS // 10))
                if c.get('falsa_apos_turno') and est is a and b and a.saudado:
                    # Uma marca de quadro falsa logo depois de cada turno de A,
                    # seguida de lixo. Cai depois do último quadro lido, antes
                    # de B fechar o turno.
                    q = E.MOLDES['quadro_longo' if b.esforco else 'quadro']
                    y = canal.passa(x)
                    t_f = lat + len(y) + int(c['falsa_apos_turno'] * FS)
                    ar_outro.soma(t_f, ab.passa(np.concatenate(
                        [0.3 * q, rng_falsa.normal(0, 0.05, FS)])))
                # O próprio alto-falante no próprio microfone, bem alto.
                ar_meu.soma(lat, 0.8 * np.tanh(x / 0.35))
        if c.get('falsas_apos_fim') and b and not injetou and getattr(a, 'encerrada', False) \
                and getattr(b, 'encerrada', False):
            # Depois da despedida: em A, uma marca de quadro seguida de lixo;
            # em B, um ACK legível de A. Nenhum dos dois pode voltar a falar
            # por isso.
            injetou = True
            q = E.MOLDES['quadro']
            ar_a.soma(t + 2 * FS, ba.passa(np.concatenate([0.3 * q, rng.normal(0, 0.05, 4 * FS)])))
            d = a.contrato_tx.descritor(a.degrau_tx, E.CTRL_N, a.modo_dados)
            ack = E.modula_quadro(d, E.empacota(E.ACK, a.sessao, bytes([a.ultimo_rx, 0]), E.CTRL_N))
            ar_b.soma(t + 2 * FS, ab.passa(ack))
        t += BLOCO
        if completo is None and b and a.aberto and b.aberto and a.confirmado \
                and b.confirmado and a.recebido == msg_b and b.recebido == msg_a \
                and not a.fila and not b.fila and a.pendente_tx is None \
                and b.pendente_tx is None:
            completo = t
        if completo is not None and (
                (getattr(a, 'encerrada', False) and getattr(b, 'encerrada', False)
                 and not a.agenda and not b.agenda
                 and t >= max(a.livre_em, b.livre_em) + DEPOIS_FIM * FS)
                or t >= completo + DEPOIS * FS):
            # Entregue não é terminado: as duas pontas ainda têm de parar de
            # falar sozinhas.
            break
    a.perdidos = perdidos
    a.apos_fim = dict(apos_fim)
    a.completo = None if completo is None else completo / FS
    a.terminou = bool(getattr(a, 'encerrada', False) and b is not None
                      and getattr(b, 'encerrada', False))
    a.sobreposicoes = [(max(f0, g0) / FS, min(f1, g1) / FS, (f0 / FS, f1 / FS), (g0 / FS, g1 / FS))
                       for f0, f1 in falas['A'] for g0, g1 in falas['B'] if min(f1, g1) > max(f0, g0)]
    a.colisao = sum(f - i for i, f, _, _ in a.sobreposicoes)
    return a, b, msg_a, msg_b, t / FS, time.time() - t0


DEPOIS = 150                     # s simulados depois de tudo entregue, para ver se param
DEPOIS_FIM = 30                  # s depois das duas encerradas, para ver se ficam caladas

CANAL_BOM = dict(perda=0.4, cauda_db=-16.0)
CENARIOS = {
    'limpo': dict(segundos=240, ruido_a=0.002, ruido_b=0.002,
                  ab=dict(CANAL_BOM), ba=dict(CANAL_BOM),
                  dados_a=120, dados_b=60),
    # A->B pior, relógios diferentes, uma faixa morta só em A->B que mata o
    # par 1700/2350 e um pedaço do 16-FSK: o rodízio de abordagens e as
    # sombras do contrato têm de dar conta.
    'adverso': dict(segundos=400, ruido_a=0.003, ruido_b=0.006,
                    ab=dict(perda=0.25, cauda_db=-12.0, relogio=1e-4,
                            morta=(1550, 1850), ecos=4),
                    ba=dict(perda=0.35, cauda_db=-14.0, relogio=-1e-4),
                    dados_a=90, dados_b=40),
    # O canal piora depois do contrato: em B aparecem apitos em cinco dos tons
    # contratados (um ventilador, uma fonte chiando) e mais ruído. O 16-FSK
    # rápido deixa de passar e as marcas continuam detectáveis -- é o caso
    # para que existe descer de degrau.
    'piora': dict(segundos=600, ruido_a=0.002, ruido_b=0.003,
                  piora_b=(62, 0.04, ((1050, 0.03), (1375, 0.03), (2025, 0.03),
                                      (2675, 0.03), (3000, 0.03))),
                  ab=dict(perda=0.3, cauda_db=-14.0, relogio=5e-5),
                  ba=dict(CANAL_BOM), dados_a=150, dados_b=20),
    # Como 'limpo', com 0,7 s de latência de saída que ninguém declarou. A
    # própria marca de fechamento sai da janela de cegueira se a guarda não
    # cobrir isso, e vira uma abertura fantasma.
    'latencia': dict(segundos=240, ruido_a=0.002, ruido_b=0.002,
                     ab=dict(CANAL_BOM), ba=dict(CANAL_BOM),
                     dados_a=120, dados_b=60, latencia_oculta=0.7),
    # Cada modulação fixa sozinha nos dados, sobre o canal 'limpo': mesma
    # saudação, sondagem e contrato, só os DADOS e ACK mudam. Prova que cada
    # quadro é escrito e lido de ponta a ponta pelo protocolo.
    **{f'modo_{m}': dict(segundos=600, ruido_a=0.002, ruido_b=0.002,
                         ab=dict(CANAL_BOM), ba=dict(CANAL_BOM),
                         dados_a=60, dados_b=30, modo_dados=m)
       for m in ('16fsk', '2fsk', 'votada', 'multicanal')},
    # As máquinas a alguns metros: nível 26 dB abaixo do 'limpo', cauda forte e
    # longa, e o microfone de B ruidoso. A saudação tem de sair das abordagens
    # de perto para o esforço de longe.
    'longe': dict(segundos=900, ruido_a=0.012, ruido_b=0.038,
                  ab=dict(perda=0.02, cauda_db=-3.0, cauda_s=0.15),
                  ba=dict(perda=0.02, cauda_db=-3.0, cauda_s=0.15),
                  dados_a=60, dados_b=30),
    # A entrada de B perdendo blocos e a saída de B com mais latência escondida
    # que a de A. O relógio de B anda atrás do ar e um quadro longo perde pedaços
    # no meio.
    'perdas': dict(segundos=400, ruido_a=0.002, ruido_b=0.004,
                   ab=dict(CANAL_BOM), ba=dict(CANAL_BOM),
                   dados_a=120, dados_b=60, perde_b=0.025,
                   latencia_oculta=0.1, latencia_oculta_b=0.2),
    # Perto num sentido e longe no outro: B ouve A forte (a chamada passa do
    # LIMIAR_LONGE) e A ouve B como de alguns metros. Só uma ponta se acha
    # longe pela chamada; a outra tem de subir o esforço pelo que lê no OLA,
    # senão responde de perto a quem só lê de longe.
    'assimetrico': dict(segundos=900, ruido_a=0.02, ruido_b=0.003,
                        ab=dict(CANAL_BOM),
                        ba=dict(perda=0.01, cauda_db=-3.0, cauda_s=0.15),
                        dados_a=60, dados_b=30),
    # Meio aberto: A leu o CONTRATO de B e abriu o canal; o CONTRATO de A se
    # perde e B, que lê os DADOS que vêm atrás, fica em 'aguarda_contrato'
    # enquanto A repete o mesmo turno.
    'meio_aberto': dict(segundos=400, ruido_a=0.002, ruido_b=0.002,
                        ab=dict(CANAL_BOM), ba=dict(CANAL_BOM),
                        dados_a=120, dados_b=60, perde_contrato=True),
    # Estouros de ruído nas duas pontas, ~um a cada 12 s, de 0,4 s: perdem
    # um quadro aqui e ali em qualquer fase. É o que deixa uma ponta achar o
    # canal aberto enquanto a outra não leu o contrato.
    'rajadas': dict(segundos=600, ruido_a=0.002, ruido_b=0.002,
                    ab=dict(CANAL_BOM), ba=dict(CANAL_BOM),
                    dados_a=120, dados_b=60, rajadas=(1 / 12, 0.4, 0.05)),
    # A latência do alto-falante Bluetooth de B na saudação (0,5 s) fica mais
    # curta depois (0,2 s), de modo que a marca de abertura do que B toca escapa
    # da cegueira e B a lê como o turno do outro; e estouros de ruído deixam
    # ilegível um turno aqui e ali, inclusive em volta do FIM.
    'jitter_bt': dict(segundos=600, ruido_a=0.002, ruido_b=0.002,
                      ab=dict(CANAL_BOM), ba=dict(CANAL_BOM),
                      dados_a=120, dados_b=60, latencia_oculta_b=0.5,
                      latencia_b_depois=(30, 0.2), rajadas=(1 / 12, 0.4, 0.05),
                      falsas_apos_fim=True),
    # B lê o CONTRATO e os DADOS do turno de A; uma marca falsa logo depois vira
    # um segmento ilegível no fim do mesmo turno, a cada repetição. B não pode
    # calar por isso.
    'marca_falsa': dict(segundos=600, ruido_a=0.002, ruido_b=0.002,
                        ab=dict(CANAL_BOM), ba=dict(CANAL_BOM),
                        dados_a=120, dados_b=60, falsa_apos_turno=0.3),
    'sem_parceiro': dict(segundos=120, ruido_a=0.01, ruido_b=0.01,
                         ab=dict(CANAL_BOM), ba=dict(CANAL_BOM),
                         sem_parceiro=True),
}


def julga(nome, a, b, msg_a, msg_b, seg, cpu):
    if CENARIOS[nome].get('sem_parceiro'):
        ok = not a.saudado and not a.aberto
        print(f"  {nome}: {seg:.0f}s simulados ({cpu:.0f}s cpu); saudado={a.saudado} "
              f"-> {'ok' if ok else 'FALHOU: achou parceiro onde não há'}")
        return ok
    entregue = a.completo is not None
    # Falar junto é falha em si: a regra é que o ouvinte nunca fala quando
    # o chamador pode falar. E entregar sem parar também: as duas pontas
    # têm de se despedir e calar.
    falou_depois = a.apos_fim['A'] > 0 or a.apos_fim['B'] > 1
    ok = entregue and a.colisao == 0 and a.terminou and not falou_depois
    porque = [] if ok else [m for m, f in (("não entregou", not entregue),
                                           ("falaram juntos", a.colisao > 0),
                                           ("não pararam", entregue and not a.terminou),
                                           (f"tocou depois do FIM {a.apos_fim}", falou_depois))
                            if f]
    print(f"  {nome}: {seg:.0f}s simulados ({cpu:.0f}s cpu); A {a.estado}, B {b.estado}; "
          f"A recebeu {len(a.recebido)}/{len(msg_b)}, B recebeu {len(b.recebido)}/{len(msg_a)}"
          f"; entregue em {a.completo if entregue else float('nan'):.0f}s"
          f" -> {'ok' if ok else 'FALHOU (' + ', '.join(porque) + ')'}"
          f" [modo de dados {a.modo_dados}]"
          f" [falando juntos {a.colisao:.1f}s]"
          + (f" [B perdeu {a.perdidos} blocos]" if a.perdidos else ""))
    for i, f, ta, tb in a.sobreposicoes[:5]:
        print(f"     juntos {i:.1f}-{f:.1f}s: A tocava {ta[0]:.1f}-{ta[1]:.1f}s, "
              f"B {tb[0]:.1f}-{tb[1]:.1f}s")
    if a.contrato_tx:
        print(f"     A->B: {a.contrato_tx.descreve()} (degrau em uso {a.degraus[a.degrau_tx]})")
    if b.contrato_tx:
        print(f"     B->A: {b.contrato_tx.descreve()} (degrau em uso {b.degraus[b.degrau_tx]})")
    return ok


def main():
    args = [x for x in sys.argv[1:] if not x.startswith('-')]
    verbose = '-v' in sys.argv
    semente = next((int(x.split('=', 1)[1]) for x in sys.argv[1:]
                    if x.startswith('--semente=')), 0)
    nomes = args or list(CENARIOS)
    ok = True
    for n in nomes:
        r = roda(n, verbose=verbose, semente=semente)
        ok &= julga(n, *r)
    print("SUCCESS!" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
