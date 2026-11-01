"""Enlace pelo ar, segunda forma: etapas em sequência, pingue-pongue estrito e
um vocabulário de sinais sonoros. Sem I/O, como `enlace.py`; `saudacao.py
--protocolo 2` liga uma `Estacao` daqui ao microfone e à caixa.

* Um sinal é uma instrução sobre a última coisa ouvida, sem bits: SOBE / FICA
  / DESCE ("o teu nível: sobe, está bom, desce"; FICA também é "ouvi/li"),
  REPETE ("ouvi um quadro e não li"), CHAMA ("estou aqui"; no refino, "a tua
  vez de tentar"). Silêncio é "não ouvi" -- é o prazo, não um sinal.
* SOBE/FICA/DESCE têm duas paridades: a cada mensagem nova quem manda alterna;
  numa repetição, mantém. Quem recebe a mesma paridade de novo sabe que a sua
  resposta se perdeu.
* Um transmite, o outro responde, uma vez. O chamador conduz todas as etapas;
  o ouvinte só responde -- nunca fala sem ter ouvido.
* Etapas: ENCONTRO (acertar o som, por assovio), REFINO (por bits: o mesmo
  bloco conhecido em três volumes, quem ouve diz DESCE/FICA/SOBE),
  TONS (a varredura da grade em cada sentido; quem ouve escolhe os 16 tons),
  VELOCIDADE (blocos conhecidos de 20 a
  100 baud, um sentido e depois o outro), DADOS (pare-e-espere).

Os quadros, as marcas de quadro, o demodulador e o CRC são os de `enlace.py`.
"""

import numpy as np
from scipy.signal import fftconvolve, sosfiltfilt

import os

import fec
import enlace
from dataclasses import replace

from enlace import (FS, Descritor, Parte, Ouvido, MOLDES, LIMIARES, QUADROS, _BANDA,
                    modula_quadro, le_quadro, conta_bits, empacota, abre, confere, q_db, MARY_TONES, chirp,
                    GANHO_FIXO, CAIXA_POS, N_POS, CAIXA_SIM, CONTRATO,
                    DADOS, FIM, DADOS_N, SONDA_N, VELOCIDADES, SONDA_V_DESC, ABORDAGENS,
                    margem_llr, GRADE, MARGEM_FORA, HUSH, LQ, mede_tons, escolhe_16, _tons_sonda,
                    _sonda_tons_nominal)

VERSAO = 16
CAIXA_INICIAL = 2              # 30%: o ouvinte começa aqui; o chamador chama de 20% para cima
CHAMA_INICIAL = 1
# Teto da subida por silêncio no encontro: o meio da escada. Silêncio antes da
# primeira resposta não diz "fraco", diz "ninguém ainda"; daí em diante só os
# vereditos do ouvinte mexem na caixa do chamador.
CHAMA_TETO = 3
TOPO = N_POS - 1
VEL = 8                        # tipo do bloco de velocidade
REF = 9                        # tipo do bloco de refino
TONS = 10                      # quem ouviu a varredura diz os tons que quer
ACORDO_TONS = 11               # o chamador fecha as duas tabelas

# --- Sinais -------------------------------------------------------------------
#
# Varreduras longas (0,8-1,0 s) e lentas (no máximo 3500 Hz/s), em quatro
# formas: banda inteira, só a metade de baixo, só a de cima, subindo ou
# descendo. Nenhuma inclinação perto das marcas de quadro (6750 e 33750 Hz/s
# subindo), para não casar com elas. A correlação de um sinal verdadeiro não
# cresce com a duração (é a fração de energia dele na janela); o que cresce é a
# distância dos falsos, e a varredura lenta aguenta melhor o eco da sala.
SINAIS = {
    'CHAMA': chirp(FS, 600, 3400, 1.0),
    'REPETE': chirp(FS, 3400, 600, 1.0),
    'FICA0': chirp(FS, 600, 3400, 0.8),
    'FICA1': chirp(FS, 3400, 600, 0.8),
    'SOBE0': chirp(FS, 600, 1900, 0.8),
    'SOBE1': chirp(FS, 1900, 600, 0.8),
    'DESCE0': chirp(FS, 2000, 3400, 0.8),
    'DESCE1': chirp(FS, 3400, 2000, 0.8),
}
LIMIAR_SINAL = 0.17            # correlação mínima para um sinal valer


# --- Sinais de etapa ---------------------------------------------------------
#
# Mudar de etapa, para a frente ou para trás, é combinado por som e nunca
# deduzido do tráfego. Um sinal por etapa: 16 tons de 62,5 ms, frequência
# 700 + 160*((m*i + k) mod 17) Hz, um declive m por etapa. Duas retas de
# declives diferentes (mod 17) coincidem em no máximo um segmento a qualquer
# deslocamento, e um tom parado (um símbolo de 16-FSK) casa com um segmento só.
# Não há etapa 3; os números e os sinais ETAPA_k são fixos.
ETAPA_NOMES = {1: 'encontro', 2: 'refino', 3: 'tons', 4: 'velocidade', 5: 'dados', 6: 'fim'}
_DECLIVES = {1: 3, 2: 5, 3: 7, 4: 10, 5: 12, 6: 14}


def _etapa_sinal(k, nseg=16, seg=0.0625, p=17):
    n = int(seg * FS)
    r = int(0.003 * FS)
    w = np.ones(n)
    w[:r], w[-r:] = np.linspace(0, 1, r), np.linspace(1, 0, r)
    fase, out = 0.0, []
    for i in range(nseg):
        f = 700 + 160 * ((_DECLIVES[k] * i + k) % p)
        out.append(np.sin(fase + 2 * np.pi * f * np.arange(n) / FS) * w)
        fase = (fase + 2 * np.pi * f * n / FS) % (2 * np.pi)
    return np.concatenate(out)


SINAIS.update({f'ETAPA_{k}': _etapa_sinal(k) for k in ETAPA_NOMES})
LMAX_SINAL = max(len(v) for v in SINAIS.values())


def _familia(nome):
    return 'ETAPA' if nome.startswith('ETAPA_') else nome.rstrip('01')
LIMIAR_ETAPA = 0.18
# Etapa interna -> etapa combinada (o número do sinal), e por onde cada papel
# entra numa etapa acordada.
GRUPO = {'encontro': 1, 'refino_meu': 2, 'refino_dele': 2, 'refino': 2,
         'tons_ida': 3, 'tons_volta': 3, 'tons_acordo': 3, 'tons': 3,
         'vel_ida': 4, 'contrato1': 4, 'vel_volta': 4, 'contrato2': 4,
         'dados_tx': 5, 'dados_rx': 5, 'fim_tx': 5, 'fim': 6}
ENTRA_CHAMADOR = {1: 'encontro', 2: 'refino_meu', 3: 'tons_ida', 4: 'vel_ida', 5: 'dados_tx', 6: 'fim'}
ENTRA_OUVINTE = {1: 'encontro', 2: 'refino', 3: 'tons', 4: 'vel_ida', 5: 'dados_rx', 6: 'fim'}
# O que cada etapa aceita ouvir. Fora disto: o ouvinte diz em que etapa está,
# o chamador propõe de novo a sua.
_VEREDITOS_TODOS = {f'{v}{p}' for v in ('SOBE', 'FICA', 'DESCE') for p in '01'}
VOCAB_SINAIS = {1: {'CHAMA'} | _VEREDITOS_TODOS, 2: {'CHAMA'} | _VEREDITOS_TODOS,
                3: {'CHAMA', 'REPETE'} | _VEREDITOS_TODOS,
                4: _VEREDITOS_TODOS | {'REPETE'}, 5: {'FICA0', 'FICA1', 'REPETE'},
                6: set()}
VOCAB_QUADROS = {1: set(), 2: {REF}, 3: {TONS, ACORDO_TONS}, 4: {VEL, CONTRATO}, 5: {DADOS, FIM}, 6: set()}
GANHO_SINAL = GANHO_FIXO
VEREDITOS = ('SOBE', 'FICA', 'DESCE')

# --- Tempos -------------------------------------------------------------------

PASSO_VARRE = FS // 4
RESP = int(0.8 * FS)           # depois do fim do que ouvi, antes de responder
LAT_CEGO = int(0.8 * FS)       # latência a mais da saída (Bluetooth)
GUARDA = int(0.3 * FS)
ATRASO_DET = int(1.6 * FS)     # da chegada do fim de um sinal até a varredura o achar
LAT_OUTRO = int(0.8 * FS)
MARGEM = int(1.0 * FS)
ASSENTA = int(0.5 * FS)
MARCA_FRACA = 0.30            # leitura aberta por marca abaixo disto pode ser largada
# Nos dados o ciclo é o que define a taxa, então lá se espera o mínimo: um quadro
# é tentado assim que acaba (a varredura vem 1,35 s depois), um sinal esperado vale
# 0,3 s depois de achado (fora dos dados, 1,5 s, porque as melodias ETAPA_* se
# imitam), e a resposta sai depois do que o outro fica cego, com folga, para a
# abertura de 0,08 s não cair na borda da cegueira dele.
ETAPAS_DADOS = ('dados_tx', 'dados_rx', 'fim_tx')
LER_CEDO = int(0.15 * FS)
ASSENTA_DADOS = int(0.3 * FS)
RESP_DADOS = RESP + GUARDA + int(0.2 * FS)
FECHO_RAZAO = 0.35             # som depois da marca / antes dela e / a marca: abaixo, é fechamento (ver _e_fechamento)
MARCA_FORTE = 0.45            # por uma marca acima disto que chegue depois
CHEIO_MIN = 0.2                # menor quarto da janela / energia média dela (ver _cheio)
# Nas varreduras, menos: o alto-falante de celular quase não dá os graves, e o
# começo de um SOBE (600-950 Hz) chega muito abaixo da média. Nas melodias ETAPA_*
# continua 0,2, para uma falsa não passar.
CHEIO_MIN_VARREDURA = 0.02
# Sinais da mesma família que se sobrepõem: o mais forte só vale com folga
# sobre o segundo; sem folga, nada se ouviu (quem mandou repete).
FOLGA_FAMILIA = 0.04
RAZAO_FAMILIA = 1.15

# --- Níveis -------------------------------------------------------------------

PICO_ALTO = 0.5
PICO_BAIXO = 0.15
CORR_BOA = 0.30                # abaixo disto o juízo pede SOBE
SNR_DISTORCE = 20.0            # nível bom, correlação ruim e bem acima do piso: distorção

# --- Quadros ------------------------------------------------------------------

CTRL = ABORDAGENS[5]           # 16-FSK lento rep2 g0.50 marca longa, 12 B, ~9,7 s
RATES = tuple(b for b, *_ in VELOCIDADES)             # 20, 33, 50, 100: a bateria de velocidade

# Redundância nos dados. A bateria de velocidade mede o erro de bits de cada
# bloco e escolhe (velocidade, rep); o REPETE sobe a rep antes de mexer em
# velocidade ou caixa. O de 10 baud entra só como queda.
REPS = (1, 2, 4)
DRATES = (10,) + RATES                                 # índice 0 = 10 baud
_DV = ((10, enlace.LENTO_GUARDA, enlace.LENTO_PRE, 'quadro_longo'),) + tuple(
    (b, g, pre, m) for b, g, pre, m, _ in VELOCIDADES)
DADOS_DESC = {(i, r): Descritor(tuple(MARY_TONES), 4, REPS[r], DADOS_N, GANHO_FIXO,
                                f'dados {b} baud rep{REPS[r]}', b, g, pre, marca=m)
              for i, (b, g, pre, m) in enumerate(_DV) for r in range(len(REPS))}
CHAVE_DADOS = {d: c for c, d in DADOS_DESC.items()}
# Erro de bits no bloco de velocidade -> rep que o aguenta (com folga sobre
# os limites de rep1, ~13%, e rep2, ~25%). Acima de 30%, aquela velocidade não serve.
BER_REP = ((0.10, 0), (0.20, 1), (0.30, 2))
# A resposta ao bloco diz a rep, só por sinal (dentro da etapa 4):
# FICA = rep1, SOBE = "sobe a redundância", rep2; DESCE = rep4; REPETE = não serve.
VEL_SINAL = ('FICA', 'SOBE', 'DESCE')
QUEDA = (0, 1)                 # nada passou: 10 baud rep2 (rep4 a 10 baud seria 101 s por quadro)
DADOS_REPETES = 6              # falhas seguidas ao mesmo DADOS antes de propor o refino de novo (duas a mais: a soma)
CARGA_DADOS = DADOS_N - 5 - 2  # seq, n


def rep_para(ber):
    for lim, r in BER_REP:
        if ber < lim:
            return r
    return None


def sobe_rep(c):
    """Um degrau de robustez: mais rep; em rep4, 10 baud rep2 (a mesma
    duração de 20 baud rep4, com o dobro de tempo por símbolo)."""
    i, r = c
    if r < len(REPS) - 1:
        return (i, r + 1)
    if i > 0:
        return QUEDA
    return c


def escada_dados(c):
    """Os quadros que quem manda pode usar a partir do contrato c: c, c
    com uma rep a mais, e mais uma. Quem recebe lê por todos eles, sem aviso."""
    c2 = sobe_rep(c)
    return (c, c2, sobe_rep(c2))


def melhor_contrato(passou):
    """{índice em RATES: rep} -> a combinação que passa um DADOS mais
    depressa; nada passou: QUEDA."""
    if not passou:
        return QUEDA
    return min(((i + 1, r) for i, r in passou.items()), key=lambda c: DADOS_DESC[c].amostras)


def _caixa_real():
    """As posições reais da caixa para o diário: `AUDIOFSK_CAIXA` troca a
    tabela de `volume.py` sem trocar `CAIXA_POS`."""
    t = os.environ.get('AUDIOFSK_CAIXA')
    if t:
        v = tuple(float(x) for x in t.split(','))
        if len(v) == len(CAIXA_POS):
            return v
    return CAIXA_POS


CAIXA_REAL = _caixa_real()


# Refino com bits. O assovio acerta o encontro, mas não diz se os dados
# passam: a caixa alta distorce os tons dos dados. Então o refino toca um bloco
# conhecido em três volumes seguidos (um degrau abaixo, o atual, um acima), em
# 16-FSK a 20 baud, e quem ouve conta os bits errados de cada um e responde só
# por sinal: DESCE (o de baixo foi melhor), FICA (o atual), SOBE (o de cima).
# Empate fica com o mais baixo: mais baixo é mais linear.
REF_DESC = SONDA_V_DESC[0]     # 16-FSK 20 baud, 16 B, ~7,0 s
REF_GAP = int(1.2 * FS)        # silêncio entre os três blocos
REF_TROCA = int(0.9 * FS)      # onde, no silêncio, a caixa troca (Bluetooth: até 0,5 s adiantada)
REF_BER = 0.02                 # diferença de erro de bit que conta
REF_MARGEM = 1.0               # dB: desempate pela margem dos bits mais fracos
REF_VEREDITO = ('DESCE', 'FICA', 'SOBE')   # índice do bloco melhor -> sinal


# Enchimento pseudoaleatório, nunca zeros: o código convolucional de zeros é
# zero, e o quadro viraria quase um tom só. Quem recebe não lê o enchimento
# (DADOS levam n no cabeçalho; os demais têm campos fixos), então mudar o
# enchimento só pede que os dois lados o gerem igual onde o quadro inteiro é
# conhecido (REF, VEL).
_ENCHE = bytes(np.random.RandomState(7919).randint(0, 256, 256).astype(np.uint8))


def empacota2(tipo, sessao, carga, n):
    carga = bytes(carga)
    return empacota(tipo, sessao, carga + _ENCHE[len(carga):n - 5], n)


def soma_bloco(bloco, desc):
    """O bloco codificado de uma tentativa (entrelaçado, rep cópias), levado
    à sequência-base do código (taxa 1/3, antes da repetição): desentrelaça e
    soma as cópias. Assim tentativas em rep e velocidades diferentes do mesmo
    quadro se somam posição a posição -- é a mesma conta que `fec.decode` faz
    com as cópias de dentro de um quadro."""
    nbits = desc.nbytes * 8
    ncod = (nbits + fec.K - 1) * len(fec.POLYS_R13)
    total = ncod * desc.rep
    llr = np.asarray(bloco, dtype=np.float64)[:total]
    if len(llr) < total:
        return None
    deint = np.zeros(total)
    deint[fec.interleave_index(total, 16)] = llr
    return deint.reshape(desc.rep, ncod).sum(axis=0)


SOMA_MAX = 6                   # tentativas somadas no máximo


def decodifica_soma(base, nbytes):
    corpo = fec.bits_to_bytes(fec.decode_soft(base, nbytes * 8, fec.POLYS_R13))
    return corpo if confere(corpo) else None


def carga_ref(j):
    rng = np.random.RandomState(5051 + j)
    return bytes([j]) + bytes(rng.randint(0, 256, SONDA_N - 6).astype(np.uint8))


def trio_amostras():
    return 3 * REF_DESC.amostras + 2 * REF_GAP


def carga_vel(i):
    rng = np.random.RandomState(4049 + i)
    return bytes([i]) + bytes(rng.randint(0, 256, SONDA_N - 6).astype(np.uint8))


# Varredura de tons. A medida é feita na sessão, porque os tons ruins mudam entre
# sessões, e depois do refino, porque a margem depende do volume.
#
# A medição e a escolha são as de `enlace.py`. A grade para em 3487,5 Hz. Num
# canal sem tom ruim sai a tabela de sempre.
N_GRADE = 19                   # GRADE[:19] = 562,5 .. 3487,5 Hz
PADRAO = tuple(range(enlace.PADRAO_16, enlace.PADRAO_16 + 16))
VARREDURA_N = 2 * LQ + 2 * HUSH + _sonda_tons_nominal()      # ~5 s


def varredura():
    m, z = MOLDES['sonda'], np.zeros(HUSH)
    return np.concatenate([m, z, _tons_sonda(0, GANHO_FIXO), z, m])


def mapa_de(tons):
    return sum(1 << k for k in tons)


def tons_de(mapa):
    return tuple(k for k in range(N_GRADE) if mapa >> k & 1)


def escolhe_mapa(margem):
    """O mapa dos 16 tons a partir das margens; 0 = não medido: a mediana
    abaixo de MARGEM_FORA é leitura fora do lugar, não canal ruim."""
    m = np.array(margem, dtype=float)
    if np.median(m[:N_GRADE]) < MARGEM_FORA:
        return 0
    m[N_GRADE:] = -99.0
    return mapa_de(escolhe_16(m))


def tabela(mapa):
    """Os tons de um Descritor; None: a tabela de sempre."""
    if not mapa or tons_de(mapa) == PADRAO:
        return None
    return tuple(GRADE[k] for k in tons_de(mapa))


def _chave_dados(d):
    """O (velocidade, rep) de um Descritor de dados, com qualquer tabela de tons."""
    return CHAVE_DADOS.get(replace(d, tons=tuple(MARY_TONES)))


class Contrato2:
    def __init__(self, c):
        self.c = tuple(c)

    @property
    def desc(self):
        return DADOS_DESC[self.c]

    def descreve(self):
        return f"16-FSK {DRATES[self.c[0]]} baud rep{REPS[self.c[1]]}"


class Ouvido2(Ouvido):
    """O `Ouvido` de `enlace.py` (memória, cegueira, trechos), com uma
    varredura própria: os oito sinais e as duas marcas de quadro, em passos de
    0,25 s, cada um com o seu limiar.

    Barata de propósito, para não ficar atrás do tempo real: o trecho passa pela
    banda (500-3700 Hz) a 48 kHz e é dizimado por 3 (16 kHz, sem dobra: nada acima
    de 3,7 kHz sobra), uma FFT só do trecho serve aos dez moldes (espectros dos
    moldes guardados), e cada achado tem a posição e a correlação refeitas a
    48 kHz em +-3 amostras. Com `quadros` falso, as marcas de quadro nem entram
    (o chamador no encontro e no refino, que só espera sinais)."""

    PASSO = PASSO_VARRE
    DIZ = 3

    def __init__(self):
        super().__init__()
        self.moldes = dict(SINAIS)
        self.moldes.update({k: MOLDES[k] for k in QUADROS})
        self.moldes['sonda'] = MOLDES['sonda']
        self.limiar = {k: (LIMIARES[k] if k in QUADROS or k == 'sonda' else LIMIAR_ETAPA if k.startswith('ETAPA_')
                           else LIMIAR_SINAL) for k in self.moldes}
        self.lmax = max(len(m) for m in self.moldes.values())
        self.quadros = True
        self._mem = np.zeros(10 * FS, dtype=np.float32)
        self._esp = {}              # (n fft) -> {tipo: conj(espectro do molde dizimado)}
        self._norma = {k: float(np.linalg.norm(m)) for k, m in self.moldes.items()}
        self._normad = {k: float(np.linalg.norm(m[::self.DIZ])) for k, m in self.moldes.items()}

    def acrescenta(self, x):
        """Como a do `Ouvido`, sem recopiar a memória inteira a cada pedaço:
        a memória tem folga e só é realocada (e aparada a GUARDA_MAX) quando enche."""
        x = np.asarray(x, dtype=np.float32).copy()
        a = self.fim
        x[self._invalido(a, a + len(x))] = 0.0
        n = self.fim - self.base
        if n + len(x) > len(self._mem):
            guarda = self._mem[max(0, n - self.GUARDA_MAX):n]
            novo = np.zeros(max(2 * (len(guarda) + len(x)), 10 * FS), dtype=np.float32)
            novo[:len(guarda)] = guarda
            self.base += n - len(guarda)
            n = len(guarda)
            self._mem = novo
        self._mem[n:n + len(x)] = x
        self.fim += len(x)
        self.buf = self._mem[:n + len(x)]

    def _espectros(self, n):
        if n not in self._esp:
            self._esp = {n: {k: np.conj(np.fft.rfft(m[::self.DIZ], n))
                             for k, m in self.moldes.items()}}
        return self._esp[n]

    def _fino(self, x, i, m, tipo):
        """Posição e correlação a 48 kHz em volta do achado dizimado."""
        lt = len(m)
        best = (0.0, i)
        for j in range(max(0, i - self.DIZ), min(len(x) - lt, i + self.DIZ) + 1):
            seg = x[j:j + lt]
            en = float(np.linalg.norm(seg)) * self._norma[tipo]
            if en > 0:
                sc = abs(float(np.dot(seg, m))) / en
                if sc > best[0]:
                    best = (sc, j)
        return best

    def varre(self, fixas=frozenset()):
        novas = []
        L = self.lmax
        pad = FS // 10
        D = self.DIZ
        tipos = [k for k in self.moldes if self.quadros or k not in QUADROS]
        while self.fim - self.varrido >= self.PASSO + L + pad:
            r0, r1 = self.varrido, self.varrido + self.PASSO
            a = r0 - pad
            x = sosfiltfilt(_BANDA, self.trecho(a, r1 + L + pad))
            inv = np.concatenate([[0], np.cumsum(self._invalido(a, r1 + L + pad))])
            xd = x[::D]
            n = 1 << int(np.ceil(np.log2(len(xd))))
            X = np.fft.rfft(xd, n)
            esp = self._espectros(n)
            e = np.concatenate([[0.0], np.cumsum(xd * xd)])
            for tipo in tipos:
                m = self.moldes[tipo]
                lt, ltd = len(m), len(m[::D])
                nq = len(xd) - ltd + 1
                c = np.abs(np.fft.irfft(X * esp[tipo], n)[:nq])
                q = np.arange(nq)
                ruim = (inv[np.minimum(D * q + lt, len(inv) - 1)] - inv[D * q]) > 0
                en = np.sqrt(np.maximum(e[q + ltd] - e[q], 0.0)) * self._normad[tipo]
                s = np.where(ruim | (en <= 0), 0.0, c / np.maximum(en, 1e-12))
                h = ltd // 2
                # Um pouco abaixo do limiar no dizimado: a correlação refeita
                # a 48 kHz é a que decide.
                for i in np.flatnonzero(s >= 0.9 * self.limiar[tipo]):
                    if s[i] < s[max(0, i - h):i + h + 1].max():
                        continue
                    sc, j = self._fino(x, D * i, m, tipo)
                    pos = a + j
                    if r0 <= pos < r1 and sc >= self.limiar[tipo]:
                        if tipo in SINAIS and not _cheio(x[j:j + lt], minimo=(
                                CHEIO_MIN if tipo.startswith('ETAPA_') else CHEIO_MIN_VARREDURA)):
                            continue
                        novas.append((pos, tipo, sc))
            self.varrido = r1
        novas.sort()
        self.marcas = [m for m in self.marcas if m[0] >= self.base] + novas
        return novas


def _cheio(seg, partes=4, minimo=CHEIO_MIN):
    """O sinal ocupa a janela inteira? A correlação normalizada divide pela
    energia da janela, e quando o sinal de verdade cobre só uma ponta dela (o
    resto é silêncio) um único trecho que case dá ~0,25, e um sinal errado
    poderia ganhar do certo. Exige energia em todos os quartos da janela."""
    n = len(seg) // partes
    if n == 0:
        return False
    e = np.mean(seg[:n * partes].reshape(partes, n) ** 2, axis=1)
    return float(e.min()) >= minimo * float(e.mean())


class Estacao:
    """Uma ponta. `ouvir(pedaço)`, depois `passo()`; o que `passo` devolver
    toca já. O relógio é o número de amostras ouvidas."""

    def __init__(self, papel, semente=None, log=None, nome=None, modo_dados='auto'):
        assert papel in ('chamador', 'ouvinte')
        self.papel = papel
        self.lider = papel == 'chamador'
        self.nome = nome or papel
        self._log = log or (lambda s: None)
        self.ouvido = Ouvido2()
        self.sessao = 0
        self.fila = bytearray()
        self.recebido = bytearray()
        self.pendente_tx = None
        self.saudado = self.aberto = self.confirmado = self.encerrada = False
        self.contrato_tx = self.contrato_rx = None
        self.degraus = tuple(f'{b} baud' for b in DRATES)
        self.conversa = False
        self.agenda = []               # [(instante, amostras, espera_resposta, resposta_max)]
        self.tocando = []
        self._no_ar = None
        self.livre_em = 0
        self.ouvido_em = -10 * FS
        self.lat_extra = 0
        self.pedidos_caixa = []
        self.pedidos_mic = []
        self.pedidos_mic_passo = []
        self.mic_no_teto = self.mic_no_piso = False
        self.k = CHAMA_INICIAL if self.lider else CAIXA_INICIAL
        self.par_tx = 1                # paridade da minha última mensagem com paridade
        self.par_rx = None             # a da última que recebi
        self.slot_ate = None           # prazo da resposta ao que toquei
        self.esperando = None
        self._sinais = []              # sinais achados, ainda assentando
        self._lendo = None             # quadro em leitura: dict(p, marca, tentados, desde)
        self.quadro_ate = 0            # fim do último quadro lido ou dado por ilegível
        self.ult_q = None              # nível do último sinal ouvido do outro lado
        self.ultimo = None             # (oque, x, espera, resposta_max) o que mandei por último
        self.n_sil = 0                 # silêncios seguidos
        self.etapa = None
        self.somas = {}                # nbytes -> (soma dos llr-base, tentativas): ver _soma
        self.sub = None
        self._sobe_seguidos = 0
        self._sc_sobe = 0.0
        self.trocas = 0
        self.c_ab = self.c_ba = None          # (índice em DRATES, índice em REPS)
        self.base_tx = self.base_rx = None     # o contrato de cada sentido, que o REPETE muda
        self.passou = {}                       # bateria de velocidade: {índice em RATES: rep}
        self.bits_errados = self.bits_lidos = 0
        self.ult_v = 'FICA'
        self.seq_tx = 0
        self.seq_rx = None
        self._quadro_tx = None
        self.falhas = 0
        self.rodou = False
        self.ref_q = None
        self.trio = None               # os três passos de caixa do meu bloco de refino
        self.notas = {}                # refino: bloco -> (erro de bit, margem) do trio que ouço
        self.ult_w = 'FICA'
        self.k_acordo = None           # caixa acertada no encontro/refino
        self.ctrl = CTRL               # descritor dos quadros de controle (CONTRATO, FIM)
        self.n_rep = 0                 # REPETEs seguidos ao mesmo quadro
        self.meu_fim = 0               # fim do meu último som, com a latência da saída
        self.prop = None               # chamador: etapa que propus e ainda não foi aceita
        self.prop_porque = ''
        self.aceito = None             # ouvinte: etapa que aceitei, esperando o FICA que fecha
        self._junta = False            # _manda acrescenta à agenda em vez de trocar
        self.tons_tx = self.tons_rx = None     # tons de cada sentido; None: MARY_TONES
        self._varredura = None         # onde começou a varredura que estou ouvindo
        self.mapa_ida = self.mapa_volta = 0
        # O que cada etapa acertou, para `percurso`: só o que foi medido.
        self.reg = dict(k_enc=None, resposta=None, ele_disse=None, eu_disse=None, som_ber=None,
                        mic0=None, vel=dict(mando={}, chega={}), motivo_tx=None, motivo_rx=None,
                        env=0, repeticoes=0, tons_mando=None, tons_chega=None)
        self._etapa('encontro', 'início')

    # --- utilidades

    @property
    def agora(self):
        return self.ouvido.fim

    @property
    def estado(self):
        return self.etapa

    def anota(self, s):
        self._log(f"[{self.nome} {self.agora / FS:7.2f}s] {s}")

    def _etapa(self, e, porque):
        if e != self.etapa:
            self.anota(f"etapa {self.etapa or '-'} -> {e}: {porque}")
            if self.etapa == 'encontro':
                self.reg['k_enc'] = self.k
        self.etapa, self.sub, self.n_sil, self.falhas, self.n_rep = e, None, 0, 0, 0
        self.prop = self.aceito = None
        self.somas = {}                    # muda a etapa, muda o quadro esperado
        if e == 'vel_ida':
            self.saudado = True            # o estado ainda publica 'saudado': som e bits acertados
            if self.k_acordo is None:
                self.k_acordo = self.k
        elif e == 'encontro':
            self.k_acordo = None
        # No encontro só há sinais: as marcas de quadro nem entram (nenhum
        # quadro muda a etapa; só o acordo por sinais). No refino há os
        # blocos conhecidos.
        self.ouvido.quadros = GRUPO[e] != 1

    def enviar(self, dados):
        self.fila += bytes(dados)

    def _degrau(self):
        return self.c_ab[0] if self.c_ab is not None else 1

    def _d_tx(self, d):
        return replace(d, tons=self.tons_tx) if self.tons_tx else d

    def _d_rx(self, d):
        return replace(d, tons=self.tons_rx) if self.tons_rx else d

    def _caixa(self, d, porque):
        k = min(TOPO, max(0, self.k + d))
        if k != self.k:
            self.anota(f"[volume] minha caixa {CAIXA_REAL[self.k]:.0%} -> {CAIXA_REAL[k]:.0%}: {porque}")
            self.k = k
            self.pedidos_caixa.append((self.agora, k))

    def _sobe_teto(self, porque):
        """Sobe a caixa por silêncio, no máximo um degrau acima do que o
        encontro e o refino acertaram. Silêncio depois do acerto não é falta de volume."""
        if self.k_acordo is not None and self.k >= min(TOPO, self.k_acordo + 1):
            self.anota(f"[volume] fico em {CAIXA_REAL[self.k]:.0%}: no máximo um degrau "
                       f"acima do acertado ({CAIXA_REAL[self.k_acordo]:.0%}); {porque}")
            return
        self._caixa(+1, porque)

    def _mic(self, d, pico):
        self.pedidos_mic_passo.append((self.agora, d, pico))

    def _q(self, pos, tipo, sc):
        """Como o sinal chegou: correlação, rms, pico e SNR sobre o piso
        (mediana do rms em janelas de 20 ms no 0,5 s antes dele)."""
        L = len(self.ouvido.moldes[tipo])
        x = self.ouvido.trecho(pos, pos + L)
        rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
        pico = float(np.max(np.abs(x))) if len(x) else 0.0
        # Piso: percentil 20 do rms em janelas de 20 ms nos 10 s antes,
        # sem o que a cegueira zerou (o meu próprio som).
        a = self.ouvido.trecho(pos - 10 * FS, pos)
        w = int(0.02 * FS)
        r = np.sqrt(np.mean(a[:len(a) // w * w].reshape(-1, w) ** 2, axis=1))
        r = r[r > 1e-6]
        piso = float(np.percentile(r, 20)) if len(r) else 1e-4
        snr = 20 * np.log10(max(rms, 1e-9) / max(piso, 1e-9))
        return dict(sc=sc, rms=rms, pico=pico, snr=snr)

    @staticmethod
    def _desc_q(q):
        return f"correlação {q['sc']:.2f}, rms {q['rms']:.3f}, pico {q['pico']:.2f}, {q['snr']:.0f} dB sobre o piso"

    # --- o que tocar

    def _manda(self, oque, x, quando, espera=True, resposta=None, nova=True):
        """Agenda uma transmissão. `oque`: nome do sinal ou rótulo do quadro."""
        x = np.asarray(x, dtype=np.float64).view(Parte)
        x.rotulo = oque
        x.caixa_marcas = [(0, self.k)]
        resp = SINAIS['CHAMA'].size if resposta is None else resposta
        item = (max(self.agora, quando), x, espera, resp)
        if self._junta:
            self.agenda.append(item)
        else:
            self.agenda = [item]
        self.ultimo = (oque, x, espera, resp)
        self.tocando = [oque]
        self.slot_ate = None
        self.anota(f"mando {oque} (caixa {CAIXA_REAL[self.k]:.0%}) em {max(self.agora, quando) / FS:.1f}s")

    def _sinal(self, nome, quando, espera=True, resposta=None):
        self._manda(nome, SINAIS[nome] * GANHO_SINAL, quando, espera, resposta)

    def _veredito(self, v, quando, nova=True, espera=True, resposta=None):
        if nova:
            self.par_tx ^= 1
        self._sinal(f"{v}{self.par_tx}", quando, espera, resposta)

    def _quadro(self, desc, tipo, carga, quando, espera=True, resposta=None, outros=()):
        """`outros`: os quadros que o outro lado pode estar esperando. Sem a
        marca de fechamento confirmada, ele só decide depois do mais longo
        deles, e a resposta chega tarde: o prazo dela cresce disso."""
        corpo = empacota2(tipo, self.sessao, carga, desc.nbytes)
        rot = f"{self._nome_tipo(tipo)} em {desc.nome}"
        if outros:
            folga = max(0, max(d.amostras for d in outros) - desc.amostras) + FS // 2 + 2048
            resposta = (SINAIS['CHAMA'].size if resposta is None else resposta) + folga
        self._manda(rot, modula_quadro(desc, corpo), quando, espera, resposta)

    def _outros_vel(self):
        return [self.ctrl] + [self._d_tx(d) for d in SONDA_V_DESC]

    def _repete(self, porque, sobe=False):
        """Silêncio: a mesma mensagem de novo, mesma paridade."""
        if self.ultimo is None:
            return
        if sobe:
            self._sobe_teto(porque)
        oque, x, espera, resp = self.ultimo
        self.anota(f"repito {oque}: {porque}")
        marcas = getattr(x, 'caixa_marcas', None) or []
        x = x.copy().view(Parte)
        # O trio de refino leva as suas três trocas de caixa; o resto toca
        # na caixa de agora (que `_sobe_teto` pode ter mudado).
        x.rotulo, x.caixa_marcas = oque, (marcas if len(marcas) > 1 else [(0, self.k)])
        self.agenda = [(self.agora, x, espera, resp)]
        self.tocando = [oque]
        self.slot_ate = None

    # --- ciclo

    def ouvir(self, x):
        self.ouvido.acrescenta(x)

    def desloca_saida(self, n):
        if n <= 0:
            return
        if self.ouvido.cegos:
            a, b = self.ouvido.cegos[-1]
            self.ouvido.cegos[-1] = (a + n, b + n)
        self.livre_em += n
        self.meu_fim += n
        if self.slot_ate is not None:
            self.slot_ate += n
        if self._no_ar is not None:
            a, m, r = self._no_ar
            self._no_ar = (a + n, m, r)

    def passo(self):
        if not self.rodou:
            self.rodou = True
            if self.lider:
                self._sinal('CHAMA', self.agora)
        for pos, tipo, sc in self.ouvido.varre():
            if self.agora < self.livre_em or self.agenda or pos < self.meu_fim or pos < self.quadro_ate:
                continue
            if tipo == 'sonda':
                if self._varredura is None and self._ouve_varredura():
                    self._varredura = pos
                    self.anota(f"varredura em {pos / FS:.2f}s ({sc:.2f}): medindo")
                continue
            if tipo in SINAIS:
                # Um sinal forte larga uma leitura de marca fraca: a melodia ETAPA_6 imita
                # a marca de quadro, e lendo esse quadro que não existe não se ouviria o
                # aceite do fim.
                if (self._lendo is not None and self._lendo['sc'] < MARCA_FRACA
                        and sc >= MARCA_FORTE and pos > self._lendo['p']):
                    self.anota(f"marca fraca em {self._lendo['p'] / FS:.2f}s largada pelo sinal {tipo}")
                    self._lendo = None
                if self._lendo is None or pos < self._lendo['p']:
                    self._sinais.append((pos, tipo, sc))
            elif self._cands(tipo) and (self._lendo is None or (
                    pos > self._lendo['p'] and sc >= MARCA_FORTE and self._lendo['sc'] < MARCA_FRACA)):
                # Uma marca fraca (falsa, no ruído) não segura a leitura quando a de
                # verdade chega logo depois.
                if self._lendo is not None:
                    self.anota(f"marca fraca em {self._lendo['p'] / FS:.2f}s ({self._lendo['sc']:.2f}) "
                               f"largada pela forte em {pos / FS:.2f}s")
                self._lendo = dict(p=pos, marca=tipo, sc=sc, tentados=set())
                self.anota(f"marca de quadro ({tipo}, {sc:.2f}) em {pos / FS:.2f}s: lendo")
        self._assenta()
        self._le()
        self._mede_varredura()
        if (self.slot_ate is not None and self.agora > self.slot_ate and self._lendo is None
                and self._varredura is None and not self._sinais and not self.agenda):
            self.slot_ate = None
            self._evento(dict(k='silencio', fim=self.agora))
        if self.agenda and self.agenda[0][0] <= self.agora and self.agora >= self.livre_em:
            _, x, espera, resp = self.agenda.pop(0)
            a = self.agora
            n = len(x)
            self.livre_em = a + n + LAT_CEGO
            self.meu_fim = a + n + LAT_CEGO
            self.ouvido.cego(a, a + n + LAT_CEGO + GUARDA)
            self._no_ar = (a, n, x.rotulo)
            self._sinais = [s for s in self._sinais if s[0] >= a + n]
            self._lendo = None
            if espera:
                self.slot_ate = (a + n + LAT_CEGO + RESP + ATRASO_DET + LAT_OUTRO
                                 + resp + ATRASO_DET + MARGEM)
                self.esperando = f"resposta a {x.rotulo} até {self.slot_ate / FS:.1f}s"
            else:
                self.slot_ate = None
                self.esperando = None
            return x
        return None

    def _assenta(self):
        """Fica o sinal mais forte entre os que se sobrepõem, depois de ASSENTA."""
        if not self._sinais:
            return
        p0, t0 = self._sinais[0][0], self._sinais[0][1]
        # Espera o sinal mais longo caber inteiro depois do primeiro achado,
        # para que um verdadeiro que começa logo depois entre no mesmo grupo.
        curto = self.etapa in ETAPAS_DADOS and not t0.startswith('ETAPA_')
        if self.ouvido.varrido < p0 + (ASSENTA_DADOS if curto else LMAX_SINAL + ASSENTA):
            return
        grupo = [s for s in self._sinais if s[0] < p0 + (ASSENTA_DADOS if curto else LMAX_SINAL)]
        self._sinais = [s for s in self._sinais if s not in grupo]
        grupo.sort(key=lambda s: -s[2])
        pos, tipo, sc = grupo[0]
        outros = [s for s in grupo[1:] if s[1] != tipo and _familia(s[1]) == _familia(tipo)]
        if outros and _familia(tipo) == 'ETAPA':
            sc2 = outros[0][2]
            if sc - sc2 < FOLGA_FAMILIA or sc < RAZAO_FAMILIA * sc2:
                self.anota(f"  sinal ambíguo: {tipo} {sc:.2f} contra {outros[0][1]} {sc2:.2f}; "
                           f"não respondo (quem mandou repete)")
                return
        if self.agenda or self.agora < self.livre_em:
            return
        q = self._q(pos, tipo, sc)
        self.ouvido_em = self.agora
        self._evento(dict(k='sinal', nome=tipo, pos=pos, fim=pos + len(SINAIS[tipo]), q=q))

    def _cands_etapa(self):
        """Os quadros que podem estar no ar nesta etapa, e só eles: uma marca
        falsa não pode ser lida como um quadro de outra etapa."""
        e, k = self.etapa, self.ctrl
        vel = [self._d_rx(d) for d in SONDA_V_DESC]
        if self.lider:
            return {'refino_dele': [REF_DESC], 'tons_ida': [k],
                    'contrato1': [k] + vel,
                    'vel_volta': vel + [k], 'contrato2': [k] + vel,
                    'fim_tx': [k] + self._dados_cands(),
                    'dados_rx': [k] + self._dados_cands()}.get(e, [])
        if e == 'encontro':
            return []
        if e == 'refino':
            return [REF_DESC]
        c = [k]
        # Na ida o ouvinte recebe os blocos até o veredito ao do topo; daí
        # só o CONTRATO. Na volta é ele quem manda blocos: só o CONTRATO.
        if e == 'vel_ida' and self.sub != 'aguarda_contrato':
            c += vel
        if e == 'dados_rx':
            c += self._dados_cands()
        if e == 'dados_tx':
            # Mandando: o chamador só responde por sinal. Com quadros aqui, uma marca
            # falsa logo antes do FICA abriria uma leitura e o FICA seria descartado.
            c = []
        return c

    def _acabou_em(self, p, fim):
        """O som parou em `fim`? No pingue-pongue quem manda cala depois do
        quadro, então 0,1-0,4 s depois o nível cai a uma fração do corpo.
        Sem o áudio desse trecho ainda: não se sabe, e espera."""
        if self.ouvido.varrido < fim + int(0.45 * FS):
            return False
        corpo = self.ouvido.trecho(p + int(0.3 * FS), max(p + int(0.4 * FS), fim - int(0.3 * FS)))
        depois = self.ouvido.trecho(fim + int(0.1 * FS), fim + int(0.4 * FS))
        if not len(corpo) or not len(depois):
            return False
        rc, rd = np.sqrt(np.mean(corpo ** 2)), np.sqrt(np.mean(depois ** 2))
        return rd < 0.3 * rc

    def _e_fechamento(self, L):
        """A marca lida é a de fechamento de um quadro cuja abertura se perdeu?
        Antes dela há corpo e depois dela o som parou; uma abertura é o contrário.
        Sem isso, uma leitura fantasma engoliria a abertura da repetição.
        None: o áudio ainda não chegou."""
        p, lm = L['p'], next((d.lm for d in self._cands(L['marca'])), 0)
        ini, fim = p + lm + int(0.3 * FS), p + lm + int(1.8 * FS)
        if not lm or self.ouvido.varrido < fim:
            return None
        rms = lambda x: float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
        antes = rms(self.ouvido.trecho(p - int(1.7 * FS), p - int(0.3 * FS)))
        marca = rms(self.ouvido.trecho(p, p + lm))
        depois = rms(self.ouvido.trecho(ini, fim))
        return depois < FECHO_RAZAO * antes and depois < FECHO_RAZAO * marca

    def _cands(self, marca):
        return sorted({d for d in self._cands_etapa() if d.marca == marca}, key=lambda d: d.amostras)

    def _dados_cands(self):
        """Os DADOS que o outro pode mandar: a escada de rep do contrato dele."""
        if self.base_rx is None:
            return []
        return list(dict.fromkeys(self._d_rx(DADOS_DESC[c]) for c in escada_dados(self.base_rx)))

    def _maior_rx(self):
        return max([d.amostras for d in self._dados_cands()] + [self.ctrl.amostras])

    def _soma(self, d, llr):
        """Uma tentativa que não passou no CRC não se joga fora: os llr dela
        se somam aos das tentativas anteriores do mesmo quadro, e a soma é
        decodificada (combinação de Chase). Cada REPETE vira, para quem
        recebe, uma rep a mais sem quadro mais longo. Só entra tentativa cujo
        comprimento a marca de fechamento confirmou (o quadro é este, e não um
        candidato lido na velocidade errada); a soma é por tamanho de corpo e
        zera ao mudar de etapa, ao ler um quadro, e depois de SOMA_MAX."""
        if _chave_dados(d) is None and d is not self.ctrl:
            return None                    # REF e VEL: cada bloco tem conteúdo próprio
        base = soma_bloco(llr, d) if llr is not None else None
        if base is None:
            return None
        s, n = self.somas.get(d.nbytes, (None, 0))
        if s is None or n >= SOMA_MAX:
            self.somas[d.nbytes] = (base, 1)
            return None
        s, n = s + base, n + 1
        self.somas[d.nbytes] = (s, n)
        corpo = decodifica_soma(s, d.nbytes)
        if corpo is not None:
            self.anota(f"li com {n} tentativas somadas (a última em {d.nome})")
        return corpo

    def _entrega(self, L, d, corpo, llr):
        p = L['p']
        c = conta_bits(corpo, llr, d) if self.etapa in ETAPAS_DADOS else None
        if c:
            self.bits_errados += c[0]
            self.bits_lidos += c[1]
            self.anota(f"{c[0]} de {c[1]} bits corrigidos ({c[0] / c[1]:.1%}) neste quadro")
        self.somas.pop(d.nbytes, None)
        self._lendo = None
        self.quadro_ate = p + d.amostras
        x = self.ouvido.trecho(p, p + d.amostras)
        q = dict(sc=L['sc'], rms=float(np.sqrt(np.mean(x * x))), pico=float(np.max(np.abs(x))))
        self.ouvido_em = self.agora
        # Lido agora, cedo: esperar a varredura atrasaria a resposta em 1,35 s.
        fim = p + d.amostras + (RESP_DADOS - RESP if self.etapa in ETAPAS_DADOS else 0)
        self._evento(dict(k='quadro', desc=d, corpo=corpo, llr=llr, fim=fim, q=q))

    def _le(self):
        """Lê o quadro em curso com cada descritor candidato, do mais curto ao
        mais longo, assim que o áudio dele chegou; o primeiro que passa no CRC
        vale. Nenhum: ilegível."""
        L = self._lendo
        if L is None:
            return
        cands = self._cands(L['marca'])
        p = L['p']
        # Só nos dados: lá o emissor repete sozinho. No refino e na velocidade o
        # veredito "ilegível" é o que faz a etapa andar, e largar sem ele a prende.
        if 'fechamento' not in L and self.etapa in ('dados_rx', 'fim_tx'):
            f = self._e_fechamento(L)
            if f is not None:
                L['fechamento'] = f
            if f:
                self._lendo = None
                self.quadro_ate = p + cands[0].lm
                self.anota(f"marca em {p / FS:.2f}s é fechamento (a abertura se perdeu): "
                           "largo a leitura e espero a repetição")
                return
        cedo = L.setdefault('cedo', set())
        for d in cands:
            if (self.etapa in ETAPAS_DADOS and d not in L['tentados'] and d not in cedo
                    and self.ouvido.fim >= p + d.amostras + LER_CEDO):
                # Sem a marca de fechamento, que a varredura ainda não viu; se
                # não passar, a tentativa de sempre vem depois, com ela.
                cedo.add(d)
                corpo, llr = le_quadro(self.ouvido.trecho(p, p + d.amostras + LER_CEDO), 0, d, None)
                if corpo is not None:
                    self._entrega(L, d, corpo, llr)
                    return
            if d in L['tentados'] or self.ouvido.varrido < p + d.amostras + FS // 2:
                continue
            L['tentados'].add(d)
            alvo = p + d.amostras - d.lm
            fecho = [pos for pos, t, _ in self.ouvido.marcas
                     if t == d.marca and abs(pos - alvo) <= int(0.004 * d.amostras)]
            audio = self.ouvido.trecho(p, p + d.amostras + FS)
            corpo, llr = le_quadro(audio, 0, d, (fecho[0] - p) if fecho else None)
            if corpo is None and fecho:
                corpo, llr = le_quadro(audio, 0, d, None)
            if llr is not None:
                L['llr'] = (d, llr)        # sem CRC, mas com sincronismo: dá para contar bits
            if corpo is None and fecho and self._acabou_em(p, p + d.amostras):
                # A marca de fechamento deste comprimento chegou e o som parou
                # ali: o quadro é deste tamanho, e os mais longos não precisam
                # ser esperados. Só a marca não basta: os dados a 100 baud
                # imitam a marca de quadro o tempo todo.
                L['fim_conhecido'] = p + d.amostras
                L['tentados'].update(cands)
                corpo = self._soma(d, llr)
            elif (corpo is None and not fecho and self.etapa in ETAPAS_DADOS
                    and self._acabou_em(p, p + d.amostras)):
                # Sem marca de fechamento, mas o som parou aqui: quem manda já
                # calou, e esperar o maior candidato só atrasa o REPETE. Não soma:
                # o comprimento não foi confirmado.
                L['fim_conhecido'] = p + d.amostras
                L['tentados'].update(cands)
            if corpo is not None:
                self._entrega(L, d, corpo, llr)
                return
        if all(d in L['tentados'] for d in cands):
            self._lendo = None
            # Ninguém responde antes de o quadro acabar: sem a marca de
            # fechamento, o fim é o do quadro mais longo que esta etapa admite.
            todos = self._cands_etapa()
            fim = L.get('fim_conhecido') or p + (max(d.amostras for d in todos) if todos else 0)
            self.quadro_ate = fim         # a marca de fechamento não abre outro quadro
            self.anota(f"quadro ilegível (marca em {p / FS:.2f}s)")
            d, llr = L.get('llr', (None, None))
            self._evento(dict(k='ilegivel', fim=fim, desc=d, llr=llr))

    def _evento(self, ev):
        if self.encerrada:
            return
        if ev['k'] == 'sinal':
            self.anota(f"ouvi {ev['nome']} ({self._desc_q(ev['q'])})")
            if self.etapa == 'encontro':
                self.reg['resposta'] = (ev['nome'], ev['q']['sc'], ev['q']['pico'])
        elif ev['k'] == 'quadro':
            t, _, _, carga = abre(ev['corpo'])
            ev['tipo'], ev['carga'] = t, carga
            self.anota(f"li {self._nome_tipo(t)} em {ev['desc'].nome}")
        elif ev['k'] == 'silencio':
            self.anota(f"silêncio: nada respondeu a {self.ultimo[0] if self.ultimo else '-'}")
        # O microfone não mexe no meio de um trio de refino: os três blocos
        # têm de ser ouvidos do mesmo jeito para serem comparados.
        if (ev['k'] in ('sinal', 'quadro') and self.etapa != 'encontro'
                and not (ev['k'] == 'quadro' and ev['tipo'] == REF)):
            self._mic_regra(ev['q'])
        if self._acordo(ev):
            return
        h = getattr(self, f"_{'l' if self.lider else 'o'}_{self.etapa}")
        if h(ev) is False:
            self.anota(f"  (ignorado na etapa {self.etapa})")

    # --- acordo de etapa (por sinais, nunca por bits nem por dedução)

    @property
    def grupo(self):
        return GRUPO[self.etapa]

    def _pertence(self, ev):
        g = self.grupo
        if ev['k'] == 'silencio':
            return True
        if ev['k'] == 'sinal':
            return ev['nome'] in VOCAB_SINAIS[g]
        if ev['k'] == 'quadro':
            return ev['tipo'] in VOCAB_QUADROS[g]
        if ev['k'] == 'varredura':
            return g == 3
        return bool(VOCAB_QUADROS[g])        # ilegível: só onde há quadros

    @staticmethod
    def _nome_tipo(t):
        return {VEL: 'VEL', REF: 'REFINO', TONS: 'TONS', ACORDO_TONS: 'ACORDO DE TONS'}.get(t) or enlace.NOMES.get(t, t)

    def _o_que(self, ev):
        if ev['k'] == 'sinal':
            return ev['nome']
        if ev['k'] == 'quadro':
            return self._nome_tipo(ev['tipo'])
        return 'um quadro ilegível' if ev['k'] == 'ilegivel' else ev['k']

    def _propoe(self, k, porque, quando=None):
        """Chamador: quero ir para a etapa k. Só vai depois do eco do ouvinte
        e do meu FICA de fecho; sem eco, repito a proposta, nunca vou sozinho."""
        self.prop, self.prop_porque, self.n_sil = k, porque, 0
        self.anota(f"proponho etapa {k} ({ETAPA_NOMES[k]}): {porque}")
        self._sinal(f'ETAPA_{k}', self.agora if quando is None else quando)

    def _acordo(self, ev):
        """Antes de qualquer etapa: os sinais de etapa, a proposta pendente
        do chamador, o aceite pendente do ouvinte, e o que não é desta etapa.
        True: o evento foi tratado aqui."""
        g = self.grupo
        if ev['k'] == 'sinal' and ev['nome'].startswith('ETAPA_'):
            j = int(ev['nome'][-1])
            return self._ouvi_etapa(j, ev) or True
        if self.lider and self.prop is not None:
            if ev['k'] == 'silencio':
                self.n_sil += 1
                self._repete(f"a proposta da etapa {self.prop} ficou sem resposta ({self.n_sil}ª)",
                             sobe=self.n_sil >= 2)
            else:
                self.anota(f"  (ouvi {self._o_que(ev)}; espero o aceite da etapa {self.prop})")
            return True
        if not self.lider and self.aceito is not None:
            if ev['k'] == 'sinal' and ev['nome'] in ('FICA0', 'FICA1'):
                k = self.aceito
                self.par_rx = self._parse(ev['nome'])[1]
                velha = self.etapa
                self._etapa(ENTRA_OUVINTE[k], f"acordado (etapa {k}, {ETAPA_NOMES[k]})")
                self._entra_ouvinte(k, velha)
                return True
            if ev['k'] == 'silencio':
                if self.aceito == 6:
                    self.anota("aceitei o fim e o FICA de fecho não veio: encerro")
                    self.encerrada = True
                return True
            # O FICA de fecho se perdeu e o chamador já fala da etapa nova:
            # não vou para lá por dedução; digo onde estou e ele propõe de novo.
            self.anota(f"ouvi {self._o_que(ev)} fora da etapa {g} (aceitei a {self.aceito}, "
                       f"sem o FICA de fecho): digo ETAPA_{g}")
            self.aceito = None
            self._sinal(f'ETAPA_{g}', ev['fim'] + RESP, espera=False)
            return True
        if not self._pertence(ev):
            if self.lider:
                self.anota(f"ouvi {self._o_que(ev)} fora da etapa {g}: proponho de novo a etapa {g}")
                self._propoe(g, "o ouvinte falou de outra etapa", ev['fim'] + RESP)
            else:
                self.anota(f"ouvi {self._o_que(ev)} fora da etapa {g}: digo ETAPA_{g}")
                self._sinal(f'ETAPA_{g}', ev['fim'] + RESP, espera=False)
            return True
        return False

    def _ouvi_etapa(self, j, ev):
        g = self.grupo
        quando = ev['fim'] + RESP
        if self.lider:
            if self.prop is not None and j == self.prop:
                k, porque = self.prop, self.prop_porque
                self.n_eco_errado = 0
                self.anota(f"o ouvinte aceitou a etapa {k}; fecho com FICA")
                self._veredito('FICA', quando, espera=False)
                velha = self.etapa
                if k == 6:
                    self._etapa('fim', f"acordado (etapa 6, fim): {porque}")
                    self.encerrada = True
                    return
                self._junta = True
                try:
                    self._entra_chamador(k, velha, porque, quando)
                finally:
                    self._junta = False
                return
            if self.prop is not None:
                self.n_eco_errado = getattr(self, 'n_eco_errado', 0) + 1
                self.anota(f"ele ecoou ETAPA_{j} em vez de ETAPA_{self.prop} ({self.n_eco_errado}ª)")
                if self.n_eco_errado >= 3 and self.prop != g:
                    # Três ecos trocados: volto à etapa em que os dois
                    # estavam de acordo, em vez de insistir para sempre.
                    self.n_eco_errado = 0
                    self.anota(f"três ecos trocados: proponho voltar à etapa {g}, a última acordada")
                    return self._propoe(g, f"ecos trocados da etapa {self.prop}", quando)
                return self._propoe(self.prop, self.prop_porque, quando)
            if j != g:
                self.anota(f"o ouvinte está na etapa {j} e eu na {g}: proponho a {g}")
                return self._propoe(g, f"o ouvinte estava na etapa {j}", quando)
            # Na mesma etapa: ele não entendeu o que mandei; mando de novo.
            return self._repete(f"o ouvinte diz que está na etapa {j}, como eu")
        # Ouvinte: o chamador propõe; aceito sempre (ele conduz).
        if self.aceito == j:
            self._sobe_teto(f"a proposta da etapa {j} veio de novo: o meu aceite se perdeu")
        self.aceito = j
        self.anota(f"aceito etapa {j} ({ETAPA_NOMES[j]})")
        self._sinal(f'ETAPA_{j}', quando, espera=(j == 6))

    def _entra_chamador(self, k, velha, porque, quando):
        if k == 1:
            ka = self.k_acordo if self.k_acordo is not None else self.k
            self._etapa('encontro', f"acordado (etapa 1, encontro): {porque}")
            if ka != self.k:
                self.k = ka
                self.pedidos_caixa.append((self.agora, ka))
            self.trocas = 0
            return self._sinal('CHAMA', quando)
        self._etapa(ENTRA_CHAMADOR[k], f"acordado (etapa {k}, {ETAPA_NOMES[k]}): {porque}")
        # O ouvinte só entra na etapa nova depois de ouvir e achar o FICA de
        # fecho; antes disso as marcas de quadro dele estão desligadas (no
        # encontro) ou ele ainda espera o vocabulário da etapa velha. O
        # primeiro som da etapa nova espera o fim do FICA + a resposta + o
        # atraso da varredura + a latência dele, para a marca de abertura não
        # cair com as marcas desligadas.
        if k != 1:
            fica = SINAIS['FICA0'].size
            quando = max(self.agora, quando) + fica + RESP + ATRASO_DET + LAT_OUTRO
        if k == 2:
            return self._l_refino_meu(None, quando)
        if k == 3:
            self.n_sil = 0
            self.mapa_ida = self.mapa_volta = 0
            return self._toca_varredura(quando, True, self.ctrl.amostras)
        if k == 4:
            self.vi = 0
            self.passou = {}
            return self._manda_vel(0, quando)
        if k == 5:
            self.aberto = True
            return self._proximo_dado(quando)

    def _entra_ouvinte(self, k, velha):
        if k == 1:
            self.par_rx = None
            self.saudado = False
        elif k == 5:
            self.aberto = True
        elif k == 6:
            self.confirmado = True
            self.anota("fim acordado: tudo entregue nos dois sentidos")
            self.encerrada = True

    # --- juízo de um sinal ouvido (regra do microfone e da caixa)

    def _mic_regra(self, q):
        """A regra do microfone em todas as etapas (no encontro `_julga` já
        a aplica)."""
        pico = q.get('pico', 0.0)
        if pico >= PICO_ALTO and not self.mic_no_piso:
            self._mic(-1, pico)
        elif pico < PICO_BAIXO and not self.mic_no_teto:
            self._mic(+1, pico)

    def _julga(self, q):
        """SOBE/FICA/DESCE sobre o sinal ouvido. O meu microfone primeiro: ele
        acerta o nível, a caixa do outro acerta o sinal sobre o ruído."""
        pico, sc, snr = q['pico'], q['sc'], q['snr']
        if pico >= PICO_ALTO:
            if not self.mic_no_piso:
                self._mic(-1, pico)
                return 'FICA', f"pico {pico:.2f}: baixo o meu microfone"
            return 'DESCE', f"pico {pico:.2f} com o meu microfone no piso"
        if pico < PICO_BAIXO and not self.mic_no_teto:
            self._mic(+1, pico)
        if sc >= CORR_BOA:
            self._sobe_seguidos = 0
            return 'FICA', f"correlação {sc:.2f} >= {CORR_BOA}"
        if snr >= SNR_DISTORCE and pico >= PICO_BAIXO:
            return 'DESCE', f"nível bom ({snr:.0f} dB) e correlação {sc:.2f}: distorção"
        if self._sobe_seguidos >= 2 and sc <= self._sc_sobe + 0.02:
            self._sobe_seguidos = 0
            return 'FICA', f"pedi SOBE duas vezes e a correlação não melhorou ({sc:.2f}): o outro está no teto"
        if self._sobe_seguidos == 0:
            self._sc_sobe = sc
        self._sobe_seguidos += 1
        return 'SOBE', f"correlação {sc:.2f} < {CORR_BOA}"

    @staticmethod
    def _parse(nome):
        return (nome[:-1], int(nome[-1])) if nome[-1] in '01' else (nome, None)

    def _aplica(self, v):
        if v == 'SOBE':
            self._caixa(+1, "o outro pediu SOBE")
        elif v == 'DESCE':
            self._caixa(-1, "o outro pediu DESCE")

    # === CHAMADOR ================================================================

    def _l_encontro(self, ev):
        if ev['k'] == 'silencio':
            self.n_sil += 1
            if self.ultimo and self.ultimo[0] == 'CHAMA':
                if self.trocas == 0 and self.k < CHAMA_TETO:
                    self._caixa(+1, "chamada sem resposta")
                elif self.trocas == 0:
                    self.anota(f"sem resposta: fico em {CAIXA_REAL[self.k]:.0%} e repito a chamada "
                               f"(ninguém ouvindo ainda?)")
                return self._sinal('CHAMA', self.agora)
            # Depois da primeira resposta, só os vereditos mexem na caixa.
            return self._repete("sem resposta")
        if ev['k'] != 'sinal':
            return False
        v, par = self._parse(ev['nome'])
        if v not in VEREDITOS:
            return False
        self.n_sil = 0
        self.trocas += 1
        self._aplica(v)
        self.ult_q = ev['q']
        mandei = self.ultimo[0][:-1] if self.ultimo else None
        if v == 'FICA' and mandei == 'FICA':
            return self._propoe(2, "mandei FICA e ouvi FICA: o som está acertado nos dois sentidos",
                                ev['fim'] + RESP)
        w, porque = self._julga(ev['q'])
        if self.trocas >= 12 and w != 'FICA':
            w, porque = 'FICA', "12 trocas: dou o som por acertado"
        self.anota(f"o meu veredito: {w} ({porque})")
        self._veredito(w, ev['fim'] + RESP)

    # --- refino com bits: o trio de blocos conhecidos e o juízo dele

    def _toca_trio(self, quando, espera):
        """O mesmo bloco conhecido em três volumes seguidos: um degrau abaixo,
        o atual, um acima. A caixa troca no silêncio entre eles."""
        self.trio = (max(0, self.k - 1), self.k, min(TOPO, self.k + 1))
        partes, marcas, a = [], [], 0
        for j, kk in enumerate(self.trio):
            if j:
                partes.append(np.zeros(REF_GAP))
                marcas.append((a + REF_TROCA, kk))
                a += REF_GAP
            else:
                marcas.append((0, kk))
            x = modula_quadro(REF_DESC, empacota2(REF, self.sessao, carga_ref(j), REF_DESC.nbytes))
            partes.append(x)
            a += len(x)
        nivs = ', '.join(f"{CAIXA_REAL[kk]:.0%}" for kk in self.trio)
        self._manda(f"REFINO em {REF_DESC.nome} ({nivs})", np.concatenate(partes), quando, espera)
        self.ultimo[1].caixa_marcas = marcas

    def _aplica_trio(self, v):
        """O veredito sobre o meu trio: fico no passo do bloco que ele achou melhor."""
        k = self.trio[REF_VEREDITO.index(v)]
        self.anota(f"refino: ele disse {v}; a minha caixa fica em {CAIXA_REAL[k]:.0%}")
        self.reg['ele_disse'] = v
        self.k = k
        self.pedidos_caixa.append((self.agora, k))

    def _julga_bloco(self, ev):
        """Um bloco do trio que ouço. Devolve o veredito quando o terceiro
        chegou; antes, None. Lido ou não, conta bits pelo sincronismo; um
        bloco sem sincronismo vale 50% e ocupa a próxima posição."""
        llr, j = ev.get('llr'), None
        if ev['k'] == 'quadro':
            j = ev['carga'][0] if ev['carga'] else None
        if j not in (0, 1, 2) and llr is not None:
            # Ilegível: qual dos três é, pelo conteúdo que mais se parece.
            j = min(range(3), key=lambda i: self._ber_ref(llr, i)[0])
        if j is None:
            j = (max(self.notas) + 1) if self.notas else 0
        if j == 0:
            self.notas = {}
        ber, m = self._ber_ref(llr, j) if llr is not None else (0.5, -99.0)
        self.notas[j] = (ber, m)
        self.anota(f"refino: bloco {j + 1}/3 ({('um degrau abaixo', 'o atual', 'um degrau acima')[j]}): "
                   f"{ber:.1%} de bits errados, margem {m:.1f} dB")
        if j < 2:
            return None
        melhor = None
        for i in sorted(self.notas):           # do mais baixo para o mais alto
            b, mm = self.notas[i]
            if (melhor is None or b < melhor[1] - REF_BER
                    or (abs(b - melhor[1]) <= REF_BER and mm > melhor[2] + REF_MARGEM)):
                melhor = (i, b, mm)
        v = REF_VEREDITO[melhor[0]]
        self.anota(f"refino: o melhor foi o bloco {melhor[0] + 1} ({melhor[1]:.1%}, {melhor[2]:.1f} dB); "
                   f"digo {v} (empate fica com o mais baixo)")
        self.notas = {}
        self.reg['eu_disse'], self.reg['som_ber'] = v, melhor[1]
        return v

    def _ber_ref(self, llr, j):
        ref = fec.encode(empacota2(REF, self.sessao, carga_ref(j), REF_DESC.nbytes), repeat=REF_DESC.rep)
        k = min(len(ref), len(llr))
        if not k:
            return 0.5, -99.0
        return float(np.mean((llr[:k] > 0) != (ref[:k] > 0))), margem_llr(llr[:k], ref[:k])

    def _l_refino_meu(self, ev, quando=None):
        """Chamador: toco o meu trio; o ouvinte diz qual volume passou melhor."""
        if ev is None:
            return self._toca_trio(self.agora if quando is None else quando, espera=True)
        if ev['k'] == 'silencio':
            self.n_sil += 1
            if self.n_sil < 3:
                return self._repete("o trio de refino ficou sem resposta")
            self._etapa('refino_dele', "o meu trio ficou sem resposta três vezes; fico onde estava")
            return self._l_refino_dele(None)
        if ev['k'] != 'sinal':
            return False
        v, _ = self._parse(ev['nome'])
        if v not in VEREDITOS:
            return False
        self.ult_q = ev['q']
        self._aplica_trio(v)
        self._etapa('refino_dele', f"minha caixa fica em {CAIXA_REAL[self.k]:.0%}")
        return self._l_refino_dele(None, ev['fim'] + RESP)

    def _l_refino_dele(self, ev, quando=None):
        """Chamador: CHAMA = "a tua vez"; ouço o trio do ouvinte, digo o
        veredito, e ele confirma com FICA."""
        if ev is None:
            self.sub, self.notas = 'chama', {}
            return self._sinal('CHAMA', self.agora if quando is None else quando,
                               resposta=trio_amostras())
        if ev['k'] == 'silencio':
            self.n_sil += 1
            if self.n_sil < 3:
                return self._repete("refino do ouvinte sem resposta")
            return self._propoe(3, "refino do ouvinte sem resposta; sigo")
        if ev['k'] in ('quadro', 'ilegivel') and self.sub == 'chama':
            if ev['k'] == 'quadro' and ev['tipo'] != REF:
                return False
            v = self._julga_bloco(ev)
            if v is None:
                return
            self.n_sil = 0
            self.sub = 'veredito'
            return self._veredito(v, ev['fim'] + RESP)
        if ev['k'] == 'sinal' and self.sub == 'veredito' and ev['nome'] in ('FICA0', 'FICA1'):
            return self._propoe(3, "refino dos dois lados feito: meço os tons", ev['fim'] + RESP)
        return False

    # --- tons

    def _ouve_varredura(self):
        return self.etapa == 'tons_volta' or (self.etapa == 'tons' and self.sub != 'acordado')

    def _toca_varredura(self, quando, espera, resposta=None):
        self._manda('VARREDURA', varredura(), quando, espera, resposta)

    def _mede_varredura(self):
        p = self._varredura
        if p is None or self.ouvido.varrido < p + VARREDURA_N + FS // 2:
            return
        self._varredura = None
        if not self._ouve_varredura():
            return
        alvo = p + VARREDURA_N - LQ
        fecho = [q for q, t, _ in self.ouvido.marcas
                 if t == 'sonda' and q != p and abs(q - alvo) <= int(0.004 * VARREDURA_N)]
        audio = self.ouvido.trecho(p, p + VARREDURA_N + FS)
        margem = mede_tons(audio, 0, (fecho[0] - p) if fecho else None, self.sessao)
        mapa = escolhe_mapa(margem)
        fora = sorted(set(PADRAO) - set(tons_de(mapa)))
        self.anota("tons: " + ("não medidos (leitura fora do lugar)" if not mapa else
                   f"margem mediana {np.median(margem[:N_GRADE]):.1f} dB, "
                   f"excluo {[GRADE[k] for k in fora] or 'nenhum'}"))
        self.quadro_ate = max(self.quadro_ate, p + VARREDURA_N)
        self.ouvido_em = self.agora
        self._evento(dict(k='varredura', mapa=mapa, fim=p + VARREDURA_N))

    def _aplica_tons(self, tx, rx):
        self.tons_tx, self.tons_rx = tabela(tx), tabela(rx)
        self.reg['tons_mando'], self.reg['tons_chega'] = tx, rx
        self.anota(f"tons acordados: daqui {'de sempre' if not self.tons_tx else 'escolhidos'}, "
                   f"de lá {'de sempre' if not self.tons_rx else 'escolhidos'}")

    def _l_tons_ida(self, ev):
        """Chamador: toquei a varredura; o ouvinte diz os tons que quer ouvir."""
        if ev['k'] == 'quadro' and ev['tipo'] == TONS:
            self.mapa_ida = int.from_bytes(ev['carga'][:4], 'little')
            self.n_sil = 0
            self._etapa('tons_volta', "o ouvinte escolheu; agora a varredura dele")
            return self._sinal('CHAMA', ev['fim'] + RESP, resposta=VARREDURA_N)
        if ev['k'] in ('silencio', 'ilegivel'):
            self.n_sil += 1
            if self.n_sil < 3:
                return self._repete("a varredura ficou sem resposta")
            self._etapa('tons_volta', "a varredura ficou sem resposta: tons de sempre nesse sentido")
            self.n_sil = 0
            return self._sinal('CHAMA', self.agora if ev['k'] == 'silencio' else ev['fim'] + RESP,
                               resposta=VARREDURA_N)
        return False

    def _l_tons_volta(self, ev):
        """Chamador: ouço a varredura do ouvinte e fecho as duas tabelas."""
        if ev['k'] == 'varredura':
            self.mapa_volta = ev['mapa']
            return self._fecha_tons(ev['fim'] + RESP)
        if ev['k'] == 'silencio':
            self.n_sil += 1
            if self.n_sil < 3:
                return self._repete("sem a varredura do ouvinte")
            return self._fecha_tons(self.agora)
        return False

    def _fecha_tons(self, quando):
        self.n_sil = 0
        self._etapa('tons_acordo', "mando as duas tabelas")
        self._quadro(self.ctrl, ACORDO_TONS,
                     (self.mapa_ida | self.mapa_volta << N_GRADE).to_bytes(5, 'little'), quando)

    def _l_tons_acordo(self, ev):
        if ev['k'] == 'sinal' and self._parse(ev['nome'])[0] == 'FICA':
            self._aplica_tons(self.mapa_ida, self.mapa_volta)
            return self._propoe(4, "tons acordados: meço a velocidade de mim para ele", ev['fim'] + RESP)
        if ev['k'] == 'silencio' or (ev['k'] == 'sinal' and ev['nome'] == 'REPETE'):
            self.n_sil += 1
            return self._repete("o acordo de tons ficou sem resposta", sobe=self.n_sil >= 3)
        return False

    def _o_tons(self, ev):
        """Ouvinte: meço a varredura e digo o que quero; no CHAMA toco a minha;
        as tabelas só valem quando o chamador as fecha, e aí confirmo com FICA."""
        if ev['k'] == 'varredura':
            return self._quadro(self.ctrl, TONS, ev['mapa'].to_bytes(4, 'little'), ev['fim'] + RESP, espera=False)
        if ev['k'] == 'sinal' and ev['nome'] == 'CHAMA':
            return self._toca_varredura(ev['fim'] + RESP, False)
        if ev['k'] == 'quadro' and ev['tipo'] == ACORDO_TONS:
            v = int.from_bytes(ev['carga'][:5], 'little')
            ida, volta = v & ((1 << N_GRADE) - 1), v >> N_GRADE & ((1 << N_GRADE) - 1)
            nova = self.sub != 'acordado'
            self.sub = 'acordado'
            self._aplica_tons(volta, ida)
            return self._veredito('FICA', ev['fim'] + RESP, nova=nova, espera=False)
        if ev['k'] == 'ilegivel':
            return self._sinal('REPETE', ev['fim'] + RESP, espera=False)
        return False

    def _manda_vel(self, i, quando):
        d = self._d_tx(SONDA_V_DESC[i])
        self._quadro(d, VEL, carga_vel(i), quando, outros=self._outros_vel())

    def _julga_vel(self, ev, esperado):
        """Erro de bits de um bloco de velocidade, lido ou não (ilegível com
        sincronismo ainda conta bits). Devolve (índice em RATES, rep ou None, ber)."""
        llr = ev.get('llr')
        if ev['k'] == 'quadro' and ev['carga']:
            i = ev['carga'][0]
        else:
            i = esperado
        if llr is None or not (0 <= i < len(RATES)):
            return i, None, 0.5
        ber = self._ber_llr(llr, carga_vel(i))
        return i, rep_para(ber), ber

    def _l_vel_ida(self, ev):
        """Chamador mandando os blocos: o ouvinte diz, por sinal, que rep cada
        velocidade aguenta (FICA rep1, SOBE rep2, DESCE rep4) ou REPETE."""
        if ev['k'] == 'sinal':
            v, _ = self._parse(ev['nome'])
            if v in VEL_SINAL:
                r = VEL_SINAL.index(v)
                self.passou[self.vi] = r
                self.reg['vel']['mando'][RATES[self.vi]] = (r, None)
                self.anota(f"velocidade {RATES[self.vi]} baud: ele leu, aguenta rep{REPS[r]}")
                self.falhas = 0
                if self.vi == len(RATES) - 1:
                    return self._fecha_ida(ev['fim'] + RESP)
                self.vi += 1
                return self._manda_vel(self.vi, ev['fim'] + RESP)
            if v != 'REPETE':
                return False
        elif ev['k'] != 'silencio':
            return False
        self.falhas += 1
        self.reg['vel']['mando'][RATES[self.vi]] = (None, None)
        self.anota(f"velocidade {RATES[self.vi]} baud: {'não serve' if ev['k'] == 'sinal' else 'sem resposta'} "
                   f"({self.falhas}ª)")
        if self.falhas >= 2:
            return self._fecha_ida(self.agora if ev['k'] == 'silencio' else ev['fim'] + RESP)
        return self._repete("de novo a mesma velocidade")

    def _fecha_ida(self, quando):
        c = melhor_contrato(self.passou)
        self.passou = {}
        self.c_ab = self.base_tx = c
        self.contrato_tx = Contrato2(c)
        self._etapa('contrato1', f"de mim para ele: {self.contrato_tx.descreve()}; agora ele mede de lá para cá")
        self.vj = 0
        self._quadro(self.ctrl, CONTRATO, bytes([c[0], c[1], 255, 255]), quando,
                     resposta=SONDA_V_DESC[0].amostras)

    def _l_contrato1(self, ev):
        if ev['k'] == 'silencio':
            self.n_sil += 1
            return self._repete("CONTRATO sem resposta", sobe=self.n_sil >= 3)
        if ev['k'] == 'sinal' and ev['nome'] == 'REPETE':
            return self._repete("ele não leu o contrato")
        if ev['k'] in ('quadro', 'ilegivel'):
            self._etapa('vel_volta', "o ouvinte começou a mandar os blocos dele")
            return self._l_vel_volta(ev)
        return False

    def _l_vel_volta(self, ev):
        """Chamador ouvindo os blocos do ouvinte: conta os bits de cada um e
        responde FICA/SOBE/DESCE (rep1/2/4) ou REPETE."""
        if ev['k'] == 'silencio':
            self.n_sil += 1
            return self._repete("sem o próximo bloco")
        if (ev['k'] == 'quadro' and ev['tipo'] == VEL) or ev['k'] == 'ilegivel':
            j, r, ber = self._julga_vel(ev, self.vj)
            if ev['k'] == 'quadro' and j < self.vj:
                return self._veredito(self.ult_v, ev['fim'] + RESP, nova=False,
                                      resposta=SONDA_V_DESC[self.vj].amostras)
            if r is not None:
                self.passou[j] = r
                self.reg['vel']['chega'][RATES[j]] = (r, ber)
                self.anota(f"velocidade {RATES[j]} baud de lá para cá: {ber:.1%} de bits errados, "
                           f"aguenta rep{REPS[r]}")
                self.falhas = 0
                self.ult_v = VEL_SINAL[r]
                if j >= len(RATES) - 1:
                    # O último bloco também leva o seu veredito, e só depois
                    # dele o CONTRATO: o ouvinte espera um sinal.
                    return self._fecha_volta_com(self.ult_v, ev['fim'] + RESP)
                self.vj = j + 1
                return self._veredito(self.ult_v, ev['fim'] + RESP, resposta=SONDA_V_DESC[self.vj].amostras)
            self.falhas += 1
            self.reg['vel']['chega'][RATES[self.vj]] = (None, ber)
            self.anota(f"velocidade {RATES[self.vj]} baud de lá para cá: {ber:.1%} de bits errados, "
                       f"não serve ({self.falhas}ª)")
            if self.falhas >= 2:
                return self._fecha_volta_com('REPETE', ev['fim'] + RESP)
            return self._sinal('REPETE', ev['fim'] + RESP, resposta=SONDA_V_DESC[self.vj].amostras)
        if ev['k'] == 'quadro' and ev['tipo'] == CONTRATO:
            return False
        return False

    def _fecha_volta_com(self, v, quando):
        """Fim da bateria de volta: o veredito do último bloco (que o ouvinte
        usa para saber que a bateria acabou: bloco do topo, ou o 2º REPETE
        seguido) e, colado nele, o CONTRATO."""
        self.anota(f"bateria de volta acabou ({v} ao último bloco); depois do veredito, o contrato")
        if v == 'REPETE':
            self._sinal('REPETE', quando, espera=False)
        else:
            self._veredito(v, quando, espera=False)
        self._junta = True
        try:
            self._fecha_volta(quando)
        finally:
            self._junta = False

    def _fecha_volta(self, quando):
        c = melhor_contrato(self.passou)
        self.passou = {}
        self.c_ba = self.base_rx = c
        self.contrato_rx = Contrato2(c)
        self._etapa('contrato2', f"de lá para cá: {self.contrato_rx.descreve()}; mando o contrato")
        self._quadro(self.ctrl, CONTRATO, bytes([*self.c_ab, *c]), quando)

    def _l_contrato2(self, ev):
        if ev['k'] == 'silencio':
            self.n_sil += 1
            return self._repete("CONTRATO sem resposta", sobe=self.n_sil >= 3)
        if ev['k'] == 'quadro' and ev['tipo'] == VEL:
            return self._repete("ele ainda manda blocos: não leu o contrato")
        if ev['k'] == 'sinal' and self._parse(ev['nome'])[0] == 'FICA':
            return self._propoe(5, f"contrato aceito: {self.contrato_tx.descreve()} para lá, "
                                   f"{self.contrato_rx.descreve()} para cá", ev['fim'] + RESP)
        if ev['k'] == 'sinal' and ev['nome'] == 'REPETE':
            return self._repete("ele não leu o contrato")
        return False

    # --- dados (os dois lados)

    def _nivel_tx(self):
        """O quadro de dados desta tentativa, pela escada do contrato. Quem
        recebe soma as tentativas falhas (ver `_soma`), então repetir o mesmo
        quadro já é ganhar rep: 1ª a 3ª no contrato, 4ª com uma rep a mais, 5ª
        em diante com mais uma."""
        c = escada_dados(self.base_tx)
        return c[0] if self.n_rep <= 2 else (c[1] if self.n_rep == 3 else c[2])

    def _proximo_dado(self, quando):
        if self.fila:
            pedaco = bytes(self.fila[:CARGA_DADOS])
            self._quadro_tx = (self.seq_tx, pedaco)
            self.pendente_tx = self._quadro_tx
            self._quadro(self._d_tx(DADOS_DESC[self._nivel_tx()]), DADOS,
                         bytes([self.seq_tx & 255, len(pedaco)]) + pedaco, quando)
            return
        self.pendente_tx = None
        self._etapa('fim_tx', "nada mais a mandar")
        self._quadro(self.ctrl, FIM, b'', quando, resposta=self._maior_rx())

    def _dados_tx(self, ev):
        """Mandando dados: FICA avança; REPETE ou silêncio repete. Quem recebe
        soma as tentativas falhas do mesmo quadro, então as duas primeiras
        repetições são iguais (cada uma já soma uma rep lá); 3º rep a mais; 4º
        mais uma (ou, já em rep4, 10 baud); 5º a caixa um degrau abaixo do
        acertado; 6º (chamador) proponho voltar ao refino. O seq só diz ao outro se o
        quadro é novo; quem conserta bits é a rep."""
        if ev['k'] == 'sinal' and self._parse(ev['nome'])[0] == 'FICA':
            par = self._parse(ev['nome'])[1]
            if par == self.par_rx:
                return self._repete("o mesmo FICA de novo: o meu quadro se perdeu")
            self.par_rx = par
            usado = self._nivel_tx()
            if usado != self.base_tx:
                self.base_tx = usado
                self.contrato_tx = Contrato2(usado)
                self.anota(f"o quadro passou em {self.contrato_tx.descreve()}: os próximos vão assim")
            seq, pedaco = self._quadro_tx
            del self.fila[:len(pedaco)]
            self.reg['env'] += len(pedaco)
            self.seq_tx += 1
            self.n_rep = 0
            return self._proximo_dado(ev['fim'] + RESP_DADOS)
        if ev['k'] == 'silencio' or (ev['k'] == 'sinal' and ev['nome'] == 'REPETE'):
            self.n_rep += 1
            n = self.n_rep
            self.reg['repeticoes'] += 1
            quando = self.agora if ev['k'] == 'silencio' else ev['fim'] + RESP_DADOS
            o_que = 'REPETE' if ev['k'] == 'sinal' else 'silêncio'
            if n >= 3:
                self.reg['motivo_tx'] = n
            if n >= DADOS_REPETES and self.lider:
                return self._propoe(2, f"{n} falhas seguidas ao mesmo DADOS, mudando redundância, "
                                       f"velocidade e caixa; refaço o refino", quando)
            if n in (1, 2):
                self.anota(f"{o_que} {n}: mando o mesmo de novo (ele soma as tentativas)")
            elif n in (3, 4):
                self.anota(f"{o_que} {n}: subo a redundância para "
                           f"{Contrato2(self._nivel_tx()).descreve()}")
            elif n == 5:
                ka = self.k_acordo if self.k_acordo is not None else self.k
                k = max(0, ka - 1)
                self.anota(f"{o_que} 5: {Contrato2(self._nivel_tx()).descreve()} com a caixa um degrau "
                           f"abaixo do acertado ({CAIXA_REAL[k]:.0%})")
                if k != self.k:
                    self.k = k
                    self.pedidos_caixa.append((self.agora, k))
            else:
                self.anota(f"{o_que} {n}: continuo em {Contrato2(self._nivel_tx()).descreve()}")
            return self._proximo_dado(quando)
        return False

    def _dados_rx(self, ev, fim_meu):
        """Recebendo dados: DADOS novo entra; FICA para qualquer lido."""
        if ev['k'] == 'quadro' and ev['tipo'] == DADOS:
            seq, n = ev['carga'][0], ev['carga'][1]
            c = _chave_dados(ev['desc'])
            if c is not None and c != self.base_rx:
                self.base_rx = c
                self.reg['motivo_rx'] = True
                self.anota(f"o outro passou a mandar em {Contrato2(c).descreve()}")
            nova = seq != self.seq_rx
            if nova:
                self.seq_rx = seq
                self.recebido += ev['carga'][2:2 + n]
                self.anota(f"dados seq {seq}: {n} bytes (total {len(self.recebido)})")
            # O mesmo quadro de novo: o meu FICA se perdeu; repito-o com a
            # mesma paridade, senão quem manda o toma por repetição e repete.
            return self._veredito('FICA', ev['fim'] + RESP, nova=nova, espera=self.lider,
                                  resposta=self._maior_rx())
        if ev['k'] == 'quadro' and ev['tipo'] == FIM:
            return fim_meu(ev)
        if ev['k'] == 'ilegivel':
            return self._sinal('REPETE', ev['fim'] + RESP_DADOS, espera=self.lider,
                               resposta=self._maior_rx())
        if ev['k'] == 'silencio':
            self.n_sil += 1
            if self.n_sil > 4:
                self.anota("o outro lado sumiu")
                self.encerrada = True
                return
            return self._repete("sem o próximo quadro")
        return False

    def _l_dados_tx(self, ev):
        return self._dados_tx(ev)

    def _l_fim_tx(self, ev):
        if ev['k'] == 'silencio' or (ev['k'] == 'sinal' and ev['nome'] == 'REPETE'):
            self.n_sil += 1
            return self._repete("FIM sem resposta", sobe=self.n_sil >= 3)
        if ev['k'] in ('quadro', 'ilegivel'):
            self.confirmado = True
            self._etapa('dados_rx', "o ouvinte respondeu ao meu FIM: agora ele manda")
            return self._l_dados_rx(ev)
        return False

    def _l_dados_rx(self, ev):
        def fim(ev):
            self.anota("FIM do ouvinte: tudo entregue nos dois sentidos")
            self.confirmado = True
            return self._propoe(6, "tudo entregue nos dois sentidos", ev['fim'] + RESP)
        return self._dados_rx(ev, fim)

    def _l_fim(self, ev):
        if ev['k'] == 'quadro' and ev['tipo'] == FIM:
            return self._veredito('FICA', ev['fim'] + RESP, nova=False, espera=False)
        return False

    # === OUVINTE =================================================================

    def _o_encontro(self, ev):
        if ev['k'] != 'sinal':
            return False
        v, par = self._parse(ev['nome'])
        repete = False
        if v == 'CHAMA':
            if self.sub == 'respondi':
                repete = True
        elif v in VEREDITOS:
            repete = par == self.par_rx
            self.par_rx = par
            if not repete:
                self._aplica(v)
        else:
            return False
        if repete:
            self._caixa(+1, "a minha resposta se perdeu (ele repetiu)")
        self.ult_q = ev['q']
        if repete:
            w, porque = self.ult_w, "o mesmo de antes"
        else:
            w, porque = self._julga(ev['q'])
        self.anota(f"o meu veredito: {w} ({porque})")
        self.ult_w = w
        if v == 'FICA' and w == 'FICA':
            self.sub = 'fecho'
            self.ref_q = ev['q']
        else:
            self.sub = 'respondi'
        self._veredito(w, ev['fim'] + RESP, nova=not repete, espera=False)

    def _o_refino(self, ev):
        """Ouvinte: julgo o trio do chamador; no CHAMA toco o meu; no
        veredito sobre o meu, fico no passo dele e confirmo com FICA."""
        if ev['k'] in ('quadro', 'ilegivel'):
            if ev['k'] == 'quadro' and ev['tipo'] != REF:
                return False
            v = self._julga_bloco(ev)
            if v is not None:
                self._veredito(v, ev['fim'] + RESP, espera=False)
            return
        if ev['k'] != 'sinal':
            return False
        v, par = self._parse(ev['nome'])
        if v == 'CHAMA':
            if self.sub == 'toquei':
                return self._repete("o meu trio se perdeu")
            self.sub = 'toquei'
            return self._toca_trio(ev['fim'] + RESP, espera=False)
        if v not in VEREDITOS or self.sub not in ('toquei', 'ack'):
            return False
        repete = par == self.par_rx
        self.par_rx = par
        if repete or self.sub == 'ack':
            return self._repete("a minha confirmação se perdeu")
        self.ult_q = ev['q']
        self._aplica_trio(v)
        self.sub = 'ack'
        return self._veredito('FICA', ev['fim'] + RESP, espera=False)

    def _o_vel_ida(self, ev):
        """Ouvinte: conto os bits de cada bloco, lido ou não, e digo que rep
        aquela velocidade aguenta: FICA rep1, SOBE rep2, DESCE rep4, REPETE não serve."""
        if (ev['k'] == 'quadro' and ev['tipo'] == VEL) or ev['k'] == 'ilegivel':
            esperado = 0 if ev['k'] == 'quadro' else self._vel_provavel(ev)
            if self.sub == 'aguarda_contrato':
                # A bateria já acabou pelos meus vereditos: um ilegível aqui
                # é o CONTRATO estragado.
                return self._sinal('REPETE', ev['fim'] + RESP, espera=False) if ev['k'] == 'ilegivel' else False
            i, r, ber = self._julga_vel(ev, esperado)
            v = 'REPETE' if r is None else VEL_SINAL[r]
            if 0 <= i < len(RATES):
                self.reg['vel']['chega'][RATES[i]] = (r, ber)
            self.anota(f"bloco a {RATES[i] if 0 <= i < len(RATES) else '?'} baud: "
                       f"{'lido' if ev['k'] == 'quadro' else 'ilegível'}, {ber:.1%} de bits errados -> "
                       f"{v}{'' if r is None else f' (rep{REPS[r]})'}")
            # A mesma conta do chamador: a bateria acaba no bloco do topo ou
            # no 2º REPETE seguido; daí em diante só o CONTRATO vale.
            self.falhas = self.falhas + 1 if r is None else 0
            if (r is not None and i >= len(RATES) - 1) or self.falhas >= 2:
                self.sub = 'aguarda_contrato'
                self.anota("a bateria de ida acabou: espero o CONTRATO")
            if r is None:
                return self._sinal('REPETE', ev['fim'] + RESP, espera=False)
            return self._veredito(v, ev['fim'] + RESP, espera=False)
        if ev['k'] == 'quadro' and ev['tipo'] == CONTRATO:
            self.c_ab = self.base_rx = (ev['carga'][0], ev['carga'][1])
            self.contrato_rx = Contrato2(self.c_ab)
            self._etapa('vel_volta', f"de lá para cá: {self.contrato_rx.descreve()}; agora mando os meus blocos")
            self.vj = 0
            return self._quadro(self._d_tx(SONDA_V_DESC[0]), VEL, carga_vel(0), ev['fim'] + RESP,
                                outros=self._outros_vel())
        return False

    def _vel_provavel(self, ev):
        """Ilegível: qual bloco era, pelo conteúdo que mais se parece."""
        llr = ev.get('llr')
        if llr is None:
            return 0
        return min(range(len(RATES)), key=lambda i: self._ber_llr(llr, carga_vel(i)))

    def _ber_llr(self, llr, conhecido):
        ref = fec.encode(empacota2(VEL, self.sessao, conhecido, SONDA_N), repeat=SONDA_V_DESC[0].rep)
        k = min(len(ref), len(llr))
        return float(np.mean((llr[:k] > 0) != (ref[:k] > 0))) if k else 0.5

    def _o_vel_volta(self, ev):
        if ev['k'] == 'quadro' and ev['tipo'] == CONTRATO:
            if ev['carga'][2] == 255:
                return self._repete("ele repetiu o contrato: o meu bloco se perdeu")
            c = ev['carga']
            self.c_ab, self.c_ba = (c[0], c[1]), (c[2], c[3])
            self.base_rx, self.base_tx = self.c_ab, self.c_ba
            self.contrato_rx, self.contrato_tx = Contrato2(self.c_ab), Contrato2(self.c_ba)
            self.anota(f"contrato: {self.contrato_rx.descreve()} de lá, {self.contrato_tx.descreve()} daqui; "
                       f"FICA, e espero a proposta da etapa 5")
            nova = self.sub != 'contratado'
            self.sub = 'contratado'
            return self._veredito('FICA', ev['fim'] + RESP, nova=nova, espera=False)
        if self.sub == 'aguarda_contrato':
            # A bateria acabou pelos vereditos: agora só o CONTRATO vale. Um
            # quadro ilegível aqui é o CONTRATO estragado: REPETE.
            if ev['k'] == 'ilegivel':
                return self._sinal('REPETE', ev['fim'] + RESP, espera=False)
            return False
        if ev['k'] == 'sinal':
            v, par = self._parse(ev['nome'])
            if v in VEL_SINAL:
                if par == self.par_rx:
                    return self._repete("o mesmo veredito de novo: o meu bloco se perdeu")
                self.par_rx = par
                self.falhas = 0
                self.reg['vel']['mando'][RATES[self.vj]] = (VEL_SINAL.index(v), None)
                if self.vj >= len(RATES) - 1:
                    self.sub = 'aguarda_contrato'
                    self.anota("veredito ao bloco do topo: a bateria acabou; espero o CONTRATO")
                    return True
                self.vj += 1
                return self._quadro(self._d_tx(SONDA_V_DESC[self.vj]), VEL, carga_vel(self.vj), ev['fim'] + RESP,
                                    outros=self._outros_vel())
            if v == 'REPETE':
                self.falhas += 1
                if self.falhas >= 2:
                    self.sub = 'aguarda_contrato'
                    self.anota("2º REPETE seguido: a bateria acabou; espero o CONTRATO")
                    return True
                return self._repete("ele não leu o bloco")
            return False
        if ev['k'] == 'silencio':
            self.n_sil += 1
            return self._repete("sem resposta ao bloco")
        return False

    def _o_dados_rx(self, ev):
        if ev['k'] == 'quadro' and ev['tipo'] == CONTRATO:
            return self._veredito('FICA', ev['fim'] + RESP, nova=False, espera=False)

        def fim(ev):
            self._etapa('dados_tx', "FIM do chamador: agora mando os meus")
            return self._proximo_dado(ev['fim'] + RESP)
        return self._dados_rx(ev, fim)

    def _o_dados_tx(self, ev):
        if ev['k'] == 'quadro' and ev['tipo'] == FIM:
            return self._repete("ele repetiu o FIM: o meu quadro se perdeu")
        return self._dados_tx(ev)

    def _o_fim_tx(self, ev):
        # O FIM é confirmado pela proposta da etapa 6 (fim), não por um FICA.
        if ev['k'] == 'sinal' and self._parse(ev['nome'])[0] == 'FICA':
            if self._parse(ev['nome'])[1] == self.par_rx:
                return self._repete("o FICA do meu último quadro de novo: o FIM se perdeu")
            return False
        if ev['k'] == 'silencio' or (ev['k'] == 'sinal' and ev['nome'] == 'REPETE'):
            self.n_sil += 1
            if self.n_sil > 3:
                self.anota("FIM sem confirmação três vezes; encerro")
                self.encerrada = True
                return
            return self._repete("FIM sem resposta")
        return False

    # --- painel

    def percurso(self, mic=None, parada=False):
        """O caminho da sessão para as telas: o que passou, onde está, o que vem,
        com o valor que cada etapa acertou. Só o que foi medido; o resto é traço,
        e sem nenhum nome interno do protocolo (esses ficam no diário).
        `mic`: a fração do microfone, que só o runtime conhece. `parada`: a sessão
        acabou, e sem chegar ao fim a etapa em que estava falhou.

        O volume do outro lado não viaja no fio: cada ponta só sabe o seu e o
        pedido que fez sobre o dele, então ele nunca tem número aqui."""
        r = self.reg
        pc = lambda k: '—' if k is None else f"{CAIXA_REAL[k]:.0%}"
        muda = lambda a, b: a if a == b else f"{a} → {b}"
        pedido = {'DESCE': 'baixar', 'FICA': 'manter', 'SOBE': 'subir'}
        if mic is not None and r['mic0'] is None:
            r['mic0'] = mic
        k_som = self.k_acordo if self.k_acordo is not None else self.k
        origem_tx = self.c_ab if self.lider else self.c_ba
        origem_rx = self.c_ba if self.lider else self.c_ab
        curto = lambda c: f"{DRATES[c[0]]} baud, repetição {REPS[c[1]]}"

        def bateria(lado, chave, origem):
            rotulo = 'Envio' if chave == 'mando' else 'Recepção'
            pontos = r['vel'][chave]
            if not pontos:
                return dict(rotulo=rotulo, valor='—', lado=lado)
            toks = []
            for b in sorted(pontos):
                rep, ber = pontos[b]
                t = f"{b} {'falhou' if rep is None else f'×{REPS[rep]}'}"
                toks.append(t if ber is None else f"{t} ({ber:.0%})")
            esc = None
            if origem is not None and DRATES[origem[0]] in pontos:
                esc = f"{DRATES[origem[0]]} ×{REPS[origem[1]]}"
            return dict(rotulo=rotulo, valor='  '.join(toks), lado=lado, escolhido=esc)

        def regua(lado, rotulo, mapa):
            """Uma marca por tom da grade: usado, excluído (da tabela de sempre),
            reserva (nem foi preciso) ou, sem medida, não medido."""
            usados = set(tons_de(mapa)) if mapa else set()
            marcas = [dict(hz=GRADE[k], estado='nm' if not mapa else 'usado' if k in usados else
                           'excluido' if k in PADRAO else 'reserva') for k in range(N_GRADE)]
            fora = [f"{GRADE[k]:g}" for k in PADRAO if k not in usados]
            if mapa is None:
                valor = '—'
            elif not mapa:
                valor = 'Não medido'
            else:
                valor = f"Usados 16 · Excluídos {len(fora)}" + (f": {' '.join(fora)} Hz" if fora else '')
            return dict(rotulo=rotulo, valor=valor, lado=lado, regua=marcas)

        tx_agora = None
        if self.base_tx is not None:
            tx_agora = self._nivel_tx() if self.etapa == 'dados_tx' else self.base_tx
        etapas = [
            dict(id='encontro', nome='Encontro', valores=[
                dict(rotulo='Volume', valor=pc(r['k_enc'] if r['k_enc'] is not None else self.k), lado='envio'),
                dict(rotulo='Sinal', valor='—' if not r['resposta'] else f"{r['resposta'][1]:.2f}", lado='chega'),
                dict(rotulo='Pico', valor='—' if not r['resposta'] else f"{r['resposta'][2]:.2f}", lado='chega')]),
            dict(id='som', nome='Som', valores=[
                dict(rotulo='Volume', valor=muda(pc(r['k_enc']), pc(k_som)) if r['k_enc'] is not None else pc(k_som),
                     lado='envio'),
                dict(rotulo='Volume dele', valor='—' if not r['eu_disse'] else 'pedi ' + pedido[r['eu_disse']],
                     lado='chega'),
                dict(rotulo='Microfone', valor='—' if mic is None else muda(f"{r['mic0']:.0%}", f"{mic:.0%}")),
                dict(rotulo='Erros', valor='—' if r['som_ber'] is None else f"{r['som_ber']:.1%}", lado='chega')]),
            dict(id='tons', nome='Tons', valores=[
                regua('envio', 'Envio', r['tons_mando']), regua('chega', 'Recepção', r['tons_chega'])]),
            dict(id='velocidade', nome='Velocidade', valores=[
                bateria('envio', 'mando', origem_tx), bateria('chega', 'chega', origem_rx)]
                 + self._testando()),
            dict(id='contrato', nome='Contrato', valores=[
                dict(rotulo='Envio', valor='—' if origem_tx is None else curto(origem_tx), lado='envio'),
                dict(rotulo='Recepção', valor='—' if origem_rx is None else curto(origem_rx), lado='chega')]),
            dict(id='dados', nome='Dados', valores=[
                dict(rotulo='Envio', lado='envio', valor='—' if tx_agora is None else
                     muda(curto(origem_tx), curto(tx_agora))),
                dict(rotulo='Recepção', lado='chega', valor='—' if self.base_rx is None else
                     muda(curto(origem_rx), curto(self.base_rx))),
                dict(rotulo='Quadro', valor='—' if not (self.seq_tx or self.seq_rx is not None) else
                     str(self.seq_tx if self.etapa == 'dados_tx' or self.seq_rx is None else self.seq_rx)),
                dict(rotulo='Repetições', valor=str(r['repeticoes'])),
                dict(rotulo='Bits corrigidos', lado='chega', valor='—' if not self.bits_lidos else
                     f"{self.bits_errados} de {self.bits_lidos} ({self.bits_errados / self.bits_lidos:.1%})"),
                dict(rotulo='Volume', lado='envio', valor=muda(pc(k_som), pc(self.k)))]),
            dict(id='fim', nome='Fim', valores=[]),
        ]
        grupo = {'encontro': 0, 'refino': 1, 'refino_meu': 1, 'refino_dele': 1,
                 'tons': 2, 'tons_ida': 2, 'tons_volta': 2, 'tons_acordo': 2, 'vel_ida': 3,
                 'contrato1': 3, 'vel_volta': 3, 'contrato2': 4, 'dados_tx': 5, 'dados_rx': 5,
                 'fim_tx': 5, 'fim': 6}
        g = grupo.get(self.etapa, 0)
        for i, e in enumerate(etapas):
            if i < g or (g == 6 and i == 6):
                e['situacao'] = 'feita'
            elif i == g:
                e['situacao'] = 'falhou' if parada else 'agora'
                if not parada:
                    e['andamento'] = self._andamento()
            else:
                e['situacao'] = 'falta'
        return etapas

    def _testando(self):
        """A velocidade em teste agora, se esta ponta souber qual é."""
        i = {'vel_ida': getattr(self, 'vi', None) if self.lider else None,
             'vel_volta': getattr(self, 'vj', None)}.get(self.etapa)
        if i is None or not 0 <= i < len(RATES):
            return []
        return [dict(rotulo='Testando', valor=f"{RATES[i]} baud, tentativa {self.falhas + 1}")]

    def _andamento(self):
        """O que acontece agora, sem nome de mensagem: enviando ou ouvindo, e por quanto."""
        t = self.agora
        if self._no_ar is not None and t < self._no_ar[0] + self._no_ar[1] + LAT_CEGO:
            return dict(sentido='enviando', s=round((self._no_ar[0] + self._no_ar[1] + LAT_CEGO - t) / FS))
        if self.agenda:
            return dict(sentido='enviando', s=None)
        if self.slot_ate is not None and self._lendo is None:
            return dict(sentido='ouvindo', s=max(0, round((self.slot_ate - t) / FS)))
        return dict(sentido='ouvindo', s=None)

    def atividade(self):
        t = self.agora
        base = f"etapa {self.etapa}"
        if self._no_ar is not None:
            a, n, rot = self._no_ar
            if t < a + n + LAT_CEGO:
                return f"{base}: tocando {rot} ({max(0, t - a) / FS:.1f} de {n / FS:.1f} s)"
        if self.agenda:
            return f"{base}: vou tocar {self.tocando[0]} em {max(0, self.agenda[0][0] - t) / FS:.1f} s"
        if self._lendo is not None:
            return f"{base}: lendo um quadro (marca há {(t - self._lendo['p']) / FS:.0f} s)"
        if self.slot_ate is not None:
            return f"{base}: esperando {self.esperando} ({max(0, self.slot_ate - t) / FS:.0f} s)"
        return f"{base}: escutando"
