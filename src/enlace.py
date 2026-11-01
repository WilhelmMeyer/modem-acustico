"""Saudação, contrato e canal: a camada de enlace que fala só pelo áudio.

Os dois lados combinam tudo pelo ar -- quem transmite, em que modo, com que
ganho, com que `fecrep` --, sem canal auxiliar, em três fases:

1. **Saudação.** Quem chama toca uma varredura subindo ("estou aqui") e escuta;
   quem ouve responde com uma varredura descendo. A varredura é larga, nenhuma
   cova do pente a apaga, o filtro casado a comprime num pico, e não precisa de
   um único bit certo para ser reconhecida. Depois disso um quadro de controle
   com CRC (`OLA` / `OLA_OK`) confirma que não foi ruído. O quadro de controle é
   tentado em várias *abordagens* (par de tons, ganho, camada) em rodízio, para
   que uma característica local que derrube uma abordagem não derrube todas;
   quem ouve tenta todas as abordagens sobre o mesmo áudio e o CRC escolhe.

2. **Contrato.** Cada lado toca uma sondagem: tons em degraus numa grade de
   162,5 Hz (em ordem embaralhada, porque a cauda do tom anterior inflaria o
   seguinte; mediana de três) e uma escada de ganhos com um bloco 16-FSK de
   conteúdo conhecido em cada degrau. Quem *recebe* mede, e por isso quem
   recebe decide o contrato daquele sentido: os 16 tons (janela deslizante
   sobre a grade, pulando até seis "sombras"), o par para 2-FSK, o ganho (o
   menor dentro de um ponto do melhor -- se um ganho menor acerta mais, a
   cadeia está comprimindo) e o degrau de taxa. O contrato é por sentido,
   porque o SNR pode diferir entre os dois.

3. **Canal.** Quadros de dados com CRC no contrato de cada sentido, pare-e-
   espere, half-duplex. A primeira troca de dados é a confirmação do contrato.
   Falhas repetidas fazem o receptor pedir um degrau mais lento (o pedido vai
   no cabeçalho de cada quadro de volta); falhas demais devolvem à saudação.

Mesma regra de camadas do `modem.py`: nada de dispositivo, thread, arquivo ou
relógio de parede. `Estacao` recebe amostras (`ouvir`) e devolve o que tocar
(`passo`); o tempo é o número de amostras ouvidas. Por isso o mesmo código
roda sob `saudacao.py` (áudio de verdade) e sob `simula_enlace.py` (dois
canais simulados), e o simulador exercita exatamente o que vai ao ar.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt

import fec
from modem import (MaryModulator, MaryDemodulator, MARY_TONES, chirp,
                   MFSKModulator, MFSKDemodulator, MFSK_PAIRS)
from xfer import crc16

FS = 48000
BAUD = 100
SPS = FS // BAUD                      # 480 amostras por símbolo, nominal

# --- Marcas -----------------------------------------------------------------
#
# Quatro varreduras, distintas por sentido e por inclinação. A chamada e a
# resposta são longas (0,2 s) porque são o primeiro contato e precisam do maior
# ganho de processamento; as de quadro e de sondagem têm 80 ms e inclinação
# mais que o dobro, o que mantém baixa a correlação cruzada entre as famílias.
MARCAS = {
    'chamada': (600.0, 3600.0, 0.20),
    'resposta': (3600.0, 600.0, 0.20),
    'quadro': (700.0, 3400.0, 0.08),
    'sonda': (3400.0, 700.0, 0.08),
    # A dos quadros de longe (ver ABORDAGENS). Cinco vezes mais longa: a
    # correlação esperada da marca verdadeira é a mesma, mas a dos falsos cai.
    'quadro_longo': (700.0, 3400.0, 0.40),
    # "Não melhorou", no refino do som: uma por lado, porque os dois a tocam e
    # nenhuma estação age sobre a do próprio tipo. Inclinação de 4500 e 3000
    # Hz/s, longe das outras (15000 da chamada e da resposta, 6750 da de quadro
    # longa), para a correlação cruzada ficar abaixo de qualquer limiar.
    'nao_ou': (2800.0, 1000.0, 0.40),      # o ouvinte diz ao chamador
    'nao_ch': (1000.0, 2800.0, 0.60),      # o chamador diz ao ouvinte
    # "Melhorou", no refino, também com marca própria: com a resposta ou a
    # chamada no lugar dela, uma varredura ainda no ar valeria "melhorou".
    # 8700 Hz/s, entre a chamada (15000) e a de quadro longa (6750); a de
    # quadro curta e a de sondagem, as mais próximas, não tocam durante o refino.
    'sim_ou': (3300.0, 700.0, 0.30),       # o ouvinte diz ao chamador
    'sim_ch': (700.0, 3300.0, 0.30),       # o chamador diz ao ouvinte
}
MOLDES = {k: chirp(FS, *v) for k, v in MARCAS.items()}
LQ = len(MOLDES['quadro'])            # 3840
HUSH = int(0.05 * FS)                 # silêncio entre a marca e o quadro
GAP = int(0.10 * FS)                  # entre segmentos de um mesmo turno

# Correlação normalizada (0 a 1) entre o molde e o trecho que ele cobre. Não
# pico sobre a mediana da região, como `modem.find_chirp`: aquilo funciona
# sobre uma gravação que contém a varredura, e aqui a região varrida pode ser
# meio segundo de silêncio colado num quadro -- a mediana do silêncio fica
# minúscula e qualquer tom do quadro passa do limiar. A normalizada não
# depende do volume nem da vizinhança e é comparável entre moldes, o que
# resolve também qual família de varredura venceu num mesmo lugar.
#
# E sempre sobre o sinal filtrado na banda das varreduras: a correlação
# normalizada divide pela energia do trecho, e ruído fora da banda só
# aumenta o divisor. Ida e volta (sosfiltfilt) para não atrasar o pico: a
# posição da marca é o relógio do quadro.
#
# Um limiar por marca, porque os erros não custam o mesmo: uma chamada falsa
# custa uma troca (o CRC do OLA recusa), uma chamada perdida custa a sessão --
# por isso a chamada fica rente ao falso. Uma marca de quadro falsa vira um
# segmento ilegível.
# Uma chamada acima de CHAMADA_CLARA é chamada, não um quadro casando com o molde.
CHAMADA_CLARA = 0.30
LIMIARES = {'chamada': 0.17, 'resposta': 0.17, 'quadro': 0.21, 'sonda': 0.26,
            'quadro_longo': 0.18, 'nao_ou': 0.17, 'nao_ch': 0.17,
            # 0,18 nos dois "melhorou": acima do que os moldes deles dão sobre
            # chamadas e respostas de verdade.
            'sim_ou': 0.18, 'sim_ch': 0.18}
QUADROS = ('quadro', 'quadro_longo')
LIMIAR = min(LIMIARES.values())
# Abaixo do limiar, marcas de quadro guardadas para a busca dirigida. Uma marca
# fraca só vale quando a estação espera uma resposta e o prazo venceu sem
# nenhuma, e só se o quadro lido dela passar no CRC: um falso custa uma
# leitura, nunca um segmento.
LIMIAR_DIRIGIDO = 0.12
_BANDA = butter(4, (500.0, 3700.0), btype='bandpass', fs=FS, output='sos')

# --- Quadros ----------------------------------------------------------------

PRE_SYM = 48                          # preâmbulo curto: o relógio vem das marcas
IDLE_SYM = 6

# As duas camadas 5x2-FSK, com o nome que `fec` dá às suas. Quadro completo,
# não encurtado: preâmbulo alternado de 80 símbolos e cauda de 4.
CAMADAS_5X2 = {'votada': 'mfsk', 'multicanal': 'mfsk-par'}
IDLE_5X2 = 4


class Parte(np.ndarray):
    """Um segmento de áudio com o rótulo do que ele é, para o diário, e
    onde trocar o volume da caixa: `caixa_marcas` [(amostra, passo)], passo
    None de volta ao do turno. Quem toca (`saudacao.py`) é quem mexe no
    volume; a estação só diz onde."""
    rotulo = None
    caixa_marcas = None


@dataclass(frozen=True)
class Descritor:
    """Tudo que o receptor precisa para ler um quadro: alfabeto e tamanho."""
    tons: tuple
    bits: int
    rep: int
    nbytes: int
    ganho: float
    nome: str = ''
    baud: int = BAUD
    guarda: float = 0.15
    pre: int = PRE_SYM
    camada: str = 'tons'        # 'tons' (16-FSK, 2-FSK), 'votada' ou 'multicanal' (5x2)
    marca: str = 'quadro'       # ou 'quadro_longo', a dos quadros de longe

    @property
    def lm(self):
        return len(MOLDES[self.marca])

    @property
    def sps(self):
        return FS // self.baud

    @property
    def nsym(self):
        if self.camada in CAMADAS_5X2:
            # Contado por `fec.layer_frame_symbols`: preâmbulo alternado de 80, corpo, cauda de 4.
            return fec.layer_frame_symbols(CAMADAS_5X2[self.camada], self.nbytes,
                                           self.rep, npairs=len(MFSK_PAIRS))
        nb = _frame_len(self.nbytes, self.rep)
        return self.pre + -(-nb // self.bits) + IDLE_SYM

    @property
    def amostras(self):
        """Duração no ar, marcas inclusas."""
        return 2 * self.lm + 2 * HUSH + self.nsym * self.sps

    @property
    def segundos(self):
        return self.amostras / FS


_FRAME_LEN = {}


def _frame_len(nbytes, rep):
    k = (nbytes, rep)
    if k not in _FRAME_LEN:
        _FRAME_LEN[k] = len(fec.frame(bytes(nbytes), repeat=rep))
    return _FRAME_LEN[k]


def _preambulo(bits, n=PRE_SYM):
    """Valores do preâmbulo. Percorre todos os tons, não só os extremos: o
    preâmbulo aqui não ensina o relógio (as marcas ensinam), ele assenta o
    piso de cada tom, e um tom que nunca soa nunca fica fora da atualização do
    piso -- o que faz cada tom convergir no mesmo ritmo."""
    m = 1 << bits
    return [(i * 7) % m if m > 2 else i % 2 for i in range(n)]


def _valores_em_bits(vals, bits):
    out = []
    for v in vals:
        out += [(v >> j) & 1 for j in range(bits)]
    return out


def modula_quadro(desc, corpo):
    """Marca, silêncio, preâmbulo, palavra de sincronismo e bloco codificado,
    cauda, silêncio, marca. Duas marcas: a primeira dá o início, o intervalo
    entre elas dá o período do símbolo medido."""
    if len(corpo) != desc.nbytes:
        raise ValueError(f"corpo de {len(corpo)} bytes num quadro de {desc.nbytes}")
    if desc.camada in CAMADAS_5X2:
        # Os tons são os MFSK_PAIRS fixos, não os da sondagem: a camada vota
        # entre os dois membros de um par.
        par = desc.camada == 'multicanal'
        k = len(MFSK_PAIRS)
        mod = MFSKModulator(fs=FS, baud=desc.baud, parallel=par)
        corpo_bits = (fec.frame_parallel(corpo, k, repeat=desc.rep) if par
                      else fec.frame(corpo, repeat=desc.rep))
        s = np.concatenate([
            mod.modulate_bits(fec.preamble_bits('mfsk', npairs=k, parallel=par)),
            mod.modulate_bits(list(corpo_bits)),
            mod.idle(IDLE_5X2)])
    else:
        mod = MaryModulator(fs=FS, baud=desc.baud, tones=desc.tons, bits=desc.bits)
        s = np.concatenate([
            mod.modulate_bits(_valores_em_bits(_preambulo(desc.bits, desc.pre), desc.bits)),
            mod.modulate_bits(list(fec.frame(corpo, repeat=desc.rep))),
            mod.idle(IDLE_SYM)])
    assert len(s) == desc.nsym * desc.sps, (len(s), desc.nsym * desc.sps)
    m = MOLDES[desc.marca]
    z = np.zeros(HUSH)
    return np.concatenate([m, z, s, z, m]) * desc.ganho


class _Demod(MaryDemodulator):
    """O mesmo receptor, com as energias calculadas por partes real e
    imaginária. Valores idênticos (a amostra é real, então |Ps|² =
    (Re P·s)² + (Im P·s)²); o motivo é tempo: matriz complexa vezes vetor real
    pode ser ordens de grandeza mais lento que as duas reais."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._partes = [(np.ascontiguousarray(p.real), np.ascontiguousarray(p.imag))
                        for p in self.probes]

    def _energies(self, start):
        key = (self.consumed, len(self.buf), start)
        if self._e_key == key:
            return self._e_val
        seg = self.buf[start + self.guard:start + self.samples_per_tone]
        e = sum((re @ seg) ** 2 + (im @ seg) ** 2 for re, im in self._partes)
        self._e_key, self._e_val = key, e
        return e


def le_quadro(audio, p1, desc, p2=None):
    """Lê um quadro de `audio` cuja marca de abertura começa em `p1`.

    Com a marca de fechamento (`p2`) o período é medido; sem ela, nominal.
    Devolve (corpo ou None, llr do bloco codificado ou None). O CRC decide se
    o corpo vale; o llr volta mesmo quando não vale, porque a sondagem conta
    bits errados sobre um conteúdo que já conhece.
    """
    sps = desc.sps
    LQ = desc.lm
    nominal = LQ + 2 * HUSH + desc.nsym * sps
    r = 1.0 if p2 is None else (p2 - p1) / nominal
    if not 0.995 <= r <= 1.005:
        return None, None
    fim = p1 + int((LQ + 2 * HUSH + (desc.nsym + 1) * sps) * r)
    seg = np.asarray(audio[p1:fim], dtype=np.float64)
    if len(seg) < (LQ + HUSH + desc.nsym * sps) * r:
        return None, None
    if desc.camada in CAMADAS_5X2:
        # Relógio congelado nas marcas, como no 16-FSK; palavra de
        # sincronismo e Viterbi por `fec.decode_block` (votação da palavra no
        # multicanal, cópias recolhidas de cada par por `pair_map`).
        par = desc.camada == 'multicanal'
        k = len(MFSK_PAIRS)
        d = MFSKDemodulator(fs=FS, baud=desc.baud, guard=desc.guarda, parallel=par,
                            steer=False, skip=int(round((LQ + HUSH) * r)), period=sps * r)
        partes = [d.demodulate_soft(seg[i:i + 2048]) for i in range(0, len(seg), 2048)]
        partes = [q for q in partes if len(q)]
        if not partes:
            return None, None
        llr = np.concatenate(partes)
        corpo, ini = fec.decode_block(llr, desc.nbytes, desc.rep, npairs=k if par else None)
        if corpo is None:
            return None, None
        return (corpo if confere(corpo) else None), llr[ini:]
    d = _Demod(fs=FS, baud=desc.baud, tones=desc.tons, bits=desc.bits,
               guard=desc.guarda, steer=False, skip=int(round((LQ + HUSH) * r)),
               period=sps * r)
    partes = [d.demodulate_soft(seg[i:i + 2048]) for i in range(0, len(seg), 2048)]
    partes = [q for q in partes if len(q)]
    if not partes:
        return None, None
    llr = np.concatenate(partes)
    # Onde a palavra de sincronismo deve terminar, procurada só a dois
    # símbolos de distância: as marcas já disseram onde o quadro está, e uma
    # busca larga só daria ao ruído do preâmbulo a chance de ganhar.
    alvo = desc.pre * desc.bits + len(fec.SYNC)
    want = 2.0 * fec.SYNC - 1.0
    melhor, onde = -2.0, None
    for e in range(alvo - 2 * desc.bits, alvo + 2 * desc.bits + 1):
        if e - len(fec.SYNC) < 0 or e > len(llr):
            continue
        sc = float(np.dot(np.sign(llr[e - len(fec.SYNC):e]), want)) / len(fec.SYNC)
        if sc > melhor:
            melhor, onde = sc, e
    if onde is None:
        return None, None
    ncod = _frame_len(desc.nbytes, desc.rep) - len(fec.SYNC)
    bloco = llr[onde:onde + ncod]
    if len(bloco) < ncod:
        return None, bloco
    corpo = fec.decode(bloco, desc.nbytes, repeat=desc.rep)
    return (corpo if confere(corpo) else None), bloco


# --- Corpos -----------------------------------------------------------------
#
# tipo, flags, sessão, carga, CRC-16. Tamanho fixo por espécie de quadro: o
# receptor precisa do tamanho para saber onde o quadro termina, e um tamanho
# que viajasse dentro do quadro só seria lido depois de o quadro ter sido lido.

OLA, OLA_OK, CONTRATO, SONDA_G, DADOS, ACK, FIM = 1, 2, 3, 4, 5, 6, 7
NOMES = {OLA: 'OLA', OLA_OK: 'OLA_OK', CONTRATO: 'CONTRATO', SONDA_G: 'SONDA_G',
         DADOS: 'DADOS', ACK: 'ACK', FIM: 'FIM'}
CTRL_N = 12
# Versão do que vai pelo ar, no terceiro byte do OLA e do OLA_OK. Mudou o
# formato, muda a versão: as duas pontas precisam da mesma, senão leem as
# sondagens uma da outra no lugar errado, e isso parece um canal ruim. Uma
# ponta sem versão manda 0 nesse byte.
VERSAO = 8
SONDA_N = 16
DADOS_N = 40
DADOS_CARGA = DADOS_N - 5 - 4          # seq, ack, pede, len


def empacota(tipo, sessao, carga, n, mais=False):
    carga = bytes(carga)
    if len(carga) > n - 5:
        raise ValueError("carga grande demais")
    c = bytes([tipo, 1 if mais else 0, sessao & 0xFF]) + carga.ljust(n - 5, b'\0')
    k = crc16(c)
    return c + bytes([k >> 8, k & 0xFF])


def conta_bits(corpo, llr, desc):
    """Bits lidos errados antes do FEC, num quadro que passou no CRC: recodifica o
    corpo certo e compara com a decisão bruta. Devolve (errados, total) ou None."""
    if corpo is None or llr is None:
        return None
    if desc.camada == 'multicanal':
        k = len(MFSK_PAIRS)
        ref = fec.frame_parallel(corpo, k, repeat=desc.rep)[len(fec.SYNC) * k:]
        n = min(len(ref), len(llr))
    else:
        ref = fec.encode(corpo, repeat=desc.rep)
        n = len(ref)
    if n == 0 or len(llr) < n:
        return None
    return int(np.sum((np.asarray(llr[:n]) > 0) != (ref[:n] > 0))), int(n)


def confere(corpo):
    if corpo is None or len(corpo) < 5:
        return False
    k = crc16(corpo[:-2])
    return corpo[-2:] == bytes([k >> 8, k & 0xFF])


def abre(corpo):
    """(tipo, mais, sessão, carga) de um corpo que já passou no CRC."""
    return corpo[0], bool(corpo[1] & 1), corpo[2], corpo[3:-2]


# --- Abordagens de controle --------------------------------------------------
#
# O quadro de controle é lento de propósito (2-FSK, 1 bit por símbolo, ~4 s
# para 12 bytes) e é tentado em rodízio. Mudam o par de tons, o ganho e por
# fim a camada inteira, para que um lugar que derrube uma não derrube todas.
# Os tons 1700/2350 são o `fsk2`; 1212/1862 é um par alternativo de mesma
# separação; a última é o 16-FSK com repetição, para um lugar em que as duas
# frequências de um par caiam ambas em covas.

# O lento: 16-FSK a 20 baud (50 ms por símbolo), guarda de 30%, repetição 2.
# Dezesseis tons espalham o risco de um tom comido por todo o bloco, onde o
# FEC o conserta; o símbolo longo põe cinco vezes mais energia em cada decisão;
# a guarda pula o começo, onde o tom anterior ainda soa. Preâmbulo curto: o
# relógio vem das marcas.
LENTO_BAUD = 20
LENTO_GUARDA = 0.30
LENTO_PRE = 8
ABORDAGENS = (
    Descritor((1700, 2350), 1, 1, CTRL_N, 0.30, 'fsk2 1700/2350 g0.30'),
    Descritor(tuple(MARY_TONES), 4, 2, CTRL_N, 0.30, '16-FSK lento rep2 g0.30',
              LENTO_BAUD, LENTO_GUARDA, LENTO_PRE),
    Descritor((1212, 1862), 1, 1, CTRL_N, 0.30, 'fsk2 1212/1862 g0.30'),
    Descritor(tuple(MARY_TONES), 4, 2, CTRL_N, 0.30, '16-FSK rep2 g0.30'),
    Descritor((1700, 2350), 1, 1, CTRL_N, 0.12, 'fsk2 1700/2350 g0.12'),
    # De longe (índices 5 e 6). Longe falta símbolo longo e marca que se ache,
    # não mais rodízio, e volume ajuda pouco porque o alto-falante comprime.
    # Então ganho só moderado e a marca longa.
    Descritor(tuple(MARY_TONES), 4, 2, CTRL_N, 0.50, '16-FSK lento rep2 g0.50 marca longa',
              LENTO_BAUD, LENTO_GUARDA, LENTO_PRE, marca='quadro_longo'),
    Descritor(tuple(MARY_TONES), 4, 2, CTRL_N, 0.50, '16-FSK muito lento rep2 g0.50 marca longa',
              10, LENTO_GUARDA, LENTO_PRE, marca='quadro_longo'),
    # Índices 7 e 8: os degraus intermediários de símbolo também no controle,
    # com a guarda e a marca dos degraus de dados de mesmo baud (`BAUD_GUARDA`,
    # definido mais abaixo: 0,24 a 50 e 0,27 a 33).
    Descritor(tuple(MARY_TONES), 4, 2, CTRL_N, 0.50, '16-FSK 50 baud rep2 g0.50 marca longa',
              50, 0.24, LENTO_PRE, marca='quadro_longo'),
    Descritor(tuple(MARY_TONES), 4, 2, CTRL_N, 0.50, '16-FSK 33 baud rep2 g0.50 marca longa',
              33, 0.27, LENTO_PRE, marca='quadro_longo'),
)
# A escada do OLA, por velocidade de símbolo: 100 -> 50 -> 33 -> 20 -> 10 baud.
# Quem decide o degrau é o chamador, e o OLA leva a decisão, para as duas pontas
# não discordarem. A 100 baud vai o 2-FSK 1700/2350. Começa a 100 se a resposta
# chegou forte (correlação >= LIMIAR_LONGE, ou alta e bem acima do piso --
# `_alta`), senão a 20 baud. A cada degrau: falhou, a caixa sobe um passo e o
# mesmo degrau se repete; falhou de novo, desce um degrau.
OLA_ESCADA = (0, 7, 8, 5, 6)
OLA_LONGE_DESDE = 3          # índice em OLA_ESCADA a partir do qual é "longe" (<= 20 baud)
# Esforço: 0 é perto, 1 é longe (marca longa, sondagem a 20 baud). É o degrau
# em que o OLA saiu (`OLA_ESCADA`), decidido pelo chamador e levado no OLA.
# Digital fixo na chamada: quem varre o nível é a caixa (`CHAMADA_PASSOS`).
GANHOS_CHAMADA = ((0.30,), (0.50,))
RMS_ALTA, SNR_ALTA = 0.03, 20.0   # ver `Estacao._alta`
LIMIAR_LONGE = 0.42          # chamada/resposta abaixo disto: longe
SONDA_GANHO_LONGE = 0.50

# --- Sondagem e contrato -----------------------------------------------------

GRADE = tuple(562.5 + 162.5 * k for k in range(25))     # 562,5 .. 4462,5 Hz
# A escada da sondagem é de VOLUME DA CAIXA (o do sistema), não de ganho
# digital: é o volume do sistema que satura, e ganho digital nenhum tira a
# caixa do limitador. O ganho digital é fixo e cada degrau k pede à própria
# máquina a caixa no passo travado + CAIXA_REL[k] -- a trava é o volume em que
# o outro lado a ouviu na chamada, então a escada anda em volta do volume que
# já se sabe servir. O contrato leva o k nos mesmos 2 bits, e cada ponta o lê
# relativo à própria trava.
GANHO_FIXO = 0.45
# Posições do controle de volume da caixa, as mesmas nas duas máquinas
# (`volume.CAIXA` tem de ter o mesmo comprimento). Os índices daqui são
# ABSOLUTOS; o resto do protocolo fala em passos relativos a `trava`.
CAIXA_POS = (0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 1.00)   # 1,00: ver volume.CAIXA
N_POS = len(CAIXA_POS)
# Acertar o som vem antes de qualquer bit. Quem chama sobe a caixa por estes
# passos até ser respondido; quem ouve só responde a uma chamada que chegou BEM
# (`CHAMADA_BOA_RMS`), e a resposta sobe pelos mesmos passos a cada chamada que
# atende. Cada lado TRAVA a caixa no passo em que foi ouvido: o chamador ao
# ouvir a resposta (o ouvinte só responde a chamada boa), o ouvinte ao ler o
# OLA (o chamador só manda OLA a resposta boa).
CHAMADA_PASSOS = (1, 2, 3, 4, 5, 6)
CAIXA_INICIAL = CHAMADA_PASSOS[0]
# Bem ouvida: rms da varredura acima disto (`_alta`). Sem teto: saturada ainda
# é ouvida, e quem satura no passo mais baixo não tem para onde descer (avisa
# no diário).
CHAMADA_BOA_RMS = 0.02
# Correlação acima da qual uma marca conta como OUVIDA, qualquer que seja o rms.
# Com isso o rms só decide subir microfone ou caixa, nunca se respondo ou confirmo.
OUVIDO_CORR = 0.21
MIC_PICO_MAX = 0.5           # acima deste pico o microfone não sobe ao acertar o som
# Ganho do microfone, de cada receptor e só dele: a cada chamada,
# resposta e quadro ouvidos antes do contrato, leva o pico recebido para a
# janela MIC_ALVO -- abaixo, um passo acima; acima, ou saturando, um abaixo.
# Microfone mais alto NÃO melhora a relação sinal/ruído (sobe o ruído da sala
# junto); só aproveita melhor o conversor e afasta a saturação. Quem melhora
# o SNR é a caixa do outro lado, e por isso a caixa é o botão de cada sentido
# (A->B é a caixa de A; B->A a de B) e o microfone só o nível.
MIC_ALVO = (0.15, 0.50)
# A caixa, corrigida pelo relato do outro lado (OLA / OLA_OK levam como cada
# ponta ouviu a outra): o alvo é o outro lado receber pico ~0,3. Acima de
# CAIXA_PICO_ALTO, ou saturando, desce um passo por 6 dB acima; abaixo de
# CAIXA_PICO_BAIXO sobe um. A chamada só diz que fui ouvido, não em que passo
# o outro lado passou a me ouvir; os bits fecham a malha.
CAIXA_PICO_BAIXO, CAIXA_PICO_ALTO = 0.10, 0.50


def q_db(x):
    """Nível 0-1 para um byte: -dBFS em meios dB (0 = 0 dBFS, 254 = -127);
    255 = não sei."""
    if x is None or not np.isfinite(x) or x <= 0:
        return 255
    return int(min(254, max(0, round(-40 * np.log10(min(1.0, x))))))


def dq_db(b):
    return None if b >= 255 else 10 ** (-b / 40)
PICO_SATURA = 0.98
# A escada da sondagem e o campo de 2 bits do contrato: passo k = trava + k - 1,
# limitado à tabela. Uma abaixo, a travada, e duas acima -- acima porque a
# trava é o primeiro passo em que o outro ouviu, o limite de baixo, e a
# margem para cima é a sondagem que mede, não um chute na hora de travar.
# Um pouco abaixo e um pouco acima do passo já refinado pelo som: o refino
# já subiu enquanto melhorava, então +2 não se mede mais.
N_CAIXA = 3
CAIXA_REL = (-1, 0, 1)
CAIXA_NEUTRA = 1             # o relativo 0: o passo travado
# Para o simulador, que não tem fader: a posição vira fator digital, linear
# em relação aos 30% (o volume em que o simulador era calibrado).
CAIXA_SIM = tuple(p / 0.30 for p in CAIXA_POS)
GANHOS = tuple(GANHO_FIXO * CAIXA_SIM[CAIXA_INICIAL + r] for r in CAIXA_REL)   # para o diário
SONDA_TOM = 0.05
SONDA_PASSO = 0.06
SONDA_SIL = 0.30
SONDA_REPS = 3
SONDA_GANHO_TOM = 0.30
SOMBRAS_MAX = 6
# Entre os degraus da escada, silêncio largo o bastante para o volume trocar
# dentro dele: a troca vai no meio, contada pela latência de saída que a
# estação mediu no próprio eco, e o erro dessa conta cabe em ±0,3 s.
GAP_CAIXA = int(0.6 * FS)
# E o microfone de quem ESCUTA: a escada vai duas vezes, a segunda com o
# microfone a 0,6 do de partida (MIC_PASSOS em `volume.py`). Quem
# ouve troca o próprio microfone no meio deste silêncio, contado da marca de
# abertura que ele achou -- a varredura anda até ~0,9 s atrás do ar e a
# entrada tem a fila dela, daí 1,6 s. Não vai no fio: só o comprimento da
# sondagem muda, e quem toca nem sabe do microfone do outro.
GAP_MIC = int(1.6 * FS)
N_MIC = 2
# Tons 16-FSK das sondas de ganho: os de sempre. O contrato escolhe os tons
# depois, e a confirmação no contrato é o que mede a escolha.
# Perto: 4 passos de caixa x 2 de microfone.
SONDA_G_DESC = tuple(Descritor(tuple(MARY_TONES), 4, 1, SONDA_N, GANHO_FIXO,
                               f'sonda caixa {k}{" mic -8 dB" if m else ""}')
                     for m in range(N_MIC) for k in range(N_CAIXA))
SONDA_G_CAIXA = tuple(k for m in range(N_MIC) for k in range(N_CAIXA))
SONDA_G_MIC = tuple(m for m in range(N_MIC) for k in range(N_CAIXA))
# De longe a escada de ganho vai no 16-FSK lento, na mesma forma do degrau
# LENTO (20 baud, guarda 0,3, marca longa): assim o receptor mede o lento de
# fato, e a velocidade passa a ser negociada como o ganho. Só os três passos
# de cima; o primeiro fica sem sonda (0,5). Sem a volta do microfone: são ~8 s
# por sonda lenta, e de longe o que falta é sinal, que baixar o microfone não dá.
SONDA_G_LONGE = tuple(None if k == 0 else
                      Descritor(tuple(MARY_TONES), 4, 1, SONDA_N, GANHO_FIXO, f'sonda lenta caixa {k}',
                                LENTO_BAUD, LENTO_GUARDA, LENTO_PRE, marca='quadro_longo')
                      for k in range(N_CAIXA))
SONDA_G_LONGE_CAIXA = tuple(range(N_CAIXA))
# Depois da escada de volume, a bateria de velocidade (VELOCIDADES, abaixo
# dos degraus): um bloco conhecido em cada troca de tom, do mais lento ao
# mais rápido, todos na caixa travada e no microfone de partida. Ela decide o
# degrau do contrato, perto e longe iguais.


def sonda_g(longe):
    return (SONDA_G_LONGE if longe else SONDA_G_DESC) + SONDA_V_DESC


def sonda_caixas(longe):
    """Passo de caixa de cada sonda (índice paralelo a `sonda_g`)."""
    return (SONDA_G_LONGE_CAIXA if longe else SONDA_G_CAIXA) + (CAIXA_NEUTRA,) * len(SONDA_V_DESC)


def sonda_mics(longe):
    """Passo de microfone de quem escuta, em cada sonda."""
    return ((0,) * len(SONDA_G_LONGE) if longe else SONDA_G_MIC) + (0,) * len(SONDA_V_DESC)


def n_escada(longe):
    """Quantas das sondas são a escada de volume; o resto é velocidade."""
    return len(SONDA_G_LONGE if longe else SONDA_G_DESC)


def sonda_layout(longe=False):
    """(início de cada sonda da escada contado da marca de abertura, ou None
    onde não há sonda; o silêncio antes de cada uma; duração total)."""
    mics = sonda_mics(longe)
    pos = 2 * LQ + 2 * HUSH + _sonda_tons_nominal()
    inicios, vaos, ant = [], [], None
    for j, d in enumerate(sonda_g(longe)):
        if d is None:
            inicios.append(None)
            vaos.append(0)
            continue
        v = GAP_MIC if ant is not None and mics[j] != ant else GAP_CAIXA
        ant = mics[j]
        pos += v
        inicios.append(pos)
        vaos.append(v)
        pos += d.amostras
    return inicios, vaos, pos + GAP


def troca_mic(longe=False):
    """Onde, contado da marca de abertura, quem escuta troca o microfone:
    [(instante, passo)], no meio do silêncio antes da primeira sonda de cada
    passo novo. Vazio de longe."""
    inicios, vaos, _ = sonda_layout(longe)
    mics = sonda_mics(longe)
    return [(inicios[j] - vaos[j] // 2, mics[j]) for j in range(1, len(mics))
            if inicios[j] is not None and mics[j] != mics[j - 1]]


# Degraus de taxa, do mais rápido ao mais lento. Limiares de erro de bit
# (antes do FEC): taxa 1/3 suave inteira até ~13%, repetida duas vezes até
# ~25%. Aqui com folga, porque três sondas por sentido são poucas e o contrato
# pode ser renegociado para baixo, nunca o contrário sem nova sondagem.
DEGRAUS = ('16-FSK rep1', '16-FSK rep2', '16-FSK rep4', '2-FSK rep1', '16-FSK 50 baud',
           '16-FSK 33 baud', '16-FSK lento', '16-FSK muito lento')
# Entre os 100 baud e o lento (20), dois degraus de símbolo intermediários:
# 100 -> 50 -> 33 -> 20 -> 10. A guarda cresce com o símbolo como o tempo dela,
# não a fração: 0,15 a 100 baud (1,5 ms) e 0,30 a 20 baud (15 ms), e a reta
# entre as duas no tempo dá 4,9 ms a 50 baud (0,24) e 8,3 ms a 33 (0,27) -- o
# que se pula é a cauda do tom anterior, que dura em ms. Marca longa e
# preâmbulo curto, como o lento. Os índices cabem nos 3 bits do contrato; mudar
# o sentido de um deles muda o fio: as duas pontas precisam da mesma versão.
R50, R33 = 4, 5
BAUD_GUARDA = {R50: (50, 0.24), R33: (33, 0.27)}
LENTO = 6
# O de longe, para quando nem o lento passa.
MUITO_LENTO = 7
LIMIARES_BER = (0.07, 0.15, 0.25, 0.35, 0.40)
# A bateria de velocidade: (baud, guarda, preâmbulo, marca, degrau que ela
# mede), do mais lento ao mais rápido, cada bloco rep1 de SONDA_N bytes --
# a mesma forma dos dados naquele degrau, então o bloco lido inteiro é o
# degrau medido, não extrapolado. Os 100 baud medem rep1/rep2/rep4 pelo erro
# de bit (`degrau_de`); o 2-FSK e o muito lento não têm bloco (o 2-FSK é
# mais lento que o 50 baud num quadro de 40 B, e o muito lento é o que sobra
# quando nada passa).
VELOCIDADES = ((LENTO_BAUD, LENTO_GUARDA, LENTO_PRE, 'quadro_longo', LENTO),
               (33, 0.27, LENTO_PRE, 'quadro_longo', R33),
               (50, 0.24, LENTO_PRE, 'quadro_longo', R50),
               (BAUD, 0.15, PRE_SYM, 'quadro', 0))
SONDA_V_DESC = tuple(Descritor(tuple(MARY_TONES), 4, 1, SONDA_N, GANHO_FIXO, f'sonda {b} baud',
                               b, g, pre, marca=m)
                     for b, g, pre, m, _ in VELOCIDADES)
# Escolha: o degrau aceito que passa mais rápido um quadro de dados; um mais
# lento só o substitui se custar menos de 20% a mais de tempo e tiver 3 dB a
# mais de margem -- a folga vale mais que um ganho de velocidade pequeno.
VEL_GANHO_MIN = 1.20
VEL_MARGEM_MELHOR = 3.0
# Só nestes estados uma marca de sonda pode ser de verdade. Fora deles ela é
# um falso no meio de outra coisa, e lida como sondagem tiraria de cena tudo o
# que viesse durante uma sondagem inteira.
ESPERA_SONDA = ('sondando', 'aguarda_sonda', 'aguarda_contrato')
MARGEM_FORA = -3.0           # dB, mediana das margens da sondagem: abaixo disto ela foi lida fora do lugar
LIMIAR_LENTO = 0.10          # erro de bit na sonda lenta abaixo disto: o lento serve

# Modulação dos dados (DADOS e ACK do canal aberto). 'auto' é a escada acima,
# que desce do 16-FSK para o 2-FSK e o lento conforme as falhas. Os outros
# fixam uma técnica só, para medir cada uma sozinha no mesmo link: a escada
# passa a mudar só a repetição, nunca a técnica, e uma técnica que não serve
# para a sala aparece como falha em vez de ser escondida pela troca. Saudação,
# sondagem e contrato são os mesmos em todos os modos -- só os dados mudam,
# para que a comparação seja justa. As duas pontas precisam do mesmo modo; o
# código dele vai num byte livre do CONTRATO (0 em 'auto') e uma divergência
# fica escrita no diário.
MODOS_DADOS = ('auto', '16fsk', '2fsk', 'votada', 'multicanal')
DEGRAUS_MODO = {
    'auto': DEGRAUS,
    # O lento e o muito lento também: sem eles, um contrato que pede lento
    # seria cortado para o rep4 a 100 baud.
    '16fsk': ('16-FSK rep1', '16-FSK rep2', '16-FSK rep4', '16-FSK 50 baud', '16-FSK 33 baud',
              '16-FSK lento', '16-FSK muito lento'),
    '2fsk': ('2-FSK rep1', '2-FSK rep2'),
    # 40 bytes a rep1 já são 11 s de 5x2 votada; rep2 passa de 20 s.
    'votada': ('5x2 votada rep1', '5x2 votada rep2'),
    'multicanal': ('5x2 multicanal rep1', '5x2 multicanal rep2', '5x2 multicanal rep4'),
}
_REPS_MODO = {'16fsk': (1, 2, 4), '2fsk': (1, 2), 'votada': (1, 2),
              'multicanal': (1, 2, 4)}


def _ordem_sonda(sessao):
    rng = np.random.RandomState(1000 + sessao)
    ordem = [k for k in range(len(GRADE)) for _ in range(SONDA_REPS)]
    rng.shuffle(ordem)
    return ordem


def carga_sonda_g(sessao, i):
    rng = np.random.RandomState(7919 * (sessao + 1) + i)
    return bytes([i]) + bytes(rng.randint(0, 256, SONDA_N - 6).astype(np.uint8))


def _tons_sonda(sessao, ganho=SONDA_GANHO_TOM):
    """A parte de tons da sondagem, entre as duas marcas de sonda."""
    passo, ntom = int(SONDA_PASSO * FS), int(SONDA_TOM * FS)
    rampa = int(0.004 * FS)
    env = np.ones(ntom)
    r = 0.5 * (1 - np.cos(np.pi * np.arange(rampa) / rampa))
    env[:rampa], env[-rampa:] = r, r[::-1]
    t = np.arange(ntom) / FS
    partes = [np.zeros(int(SONDA_SIL * FS))]
    for k in _ordem_sonda(sessao):
        p = np.zeros(passo)
        p[:ntom] = np.sin(2 * np.pi * GRADE[k] * t) * env
        partes.append(p)
    return np.concatenate(partes) * ganho


def _sonda_tons_nominal():
    return int(SONDA_SIL * FS) + len(GRADE) * SONDA_REPS * int(SONDA_PASSO * FS)


def modula_sonda(sessao, ganho_tons=SONDA_GANHO_TOM, longe=False):
    """`ganho_tons` só muda o nível dos tons, não o que se mede com eles: a
    margem de cada tom é contra o piso e contra os outros tons, todos no
    mesmo ganho. As sondas da escada, depois, vão todas em GANHO_FIXO, cada
    uma num passo de caixa (`caixa_marcas` do Parte: onde trocar o volume)."""
    m = MOLDES['sonda']
    z = np.zeros(HUSH)
    inicios, vaos, total = sonda_layout(longe)
    partes = [m, z, _tons_sonda(sessao, ganho_tons), z, m]
    marcas = []
    for i, (d, k) in enumerate(zip(sonda_g(longe), sonda_caixas(longe))):
        if d is None:
            continue
        partes += [np.zeros(vaos[i]),
                   modula_quadro(d, empacota(SONDA_G, sessao, carga_sonda_g(sessao, i), SONDA_N))]
        marcas.append((inicios[i] - vaos[i] // 2, k))
    partes.append(np.zeros(GAP))
    x = np.concatenate(partes).view(Parte)
    assert len(x) == total
    # Depois da última sonda, de volta ao passo do turno; um pouco além do
    # fim, para não cortar o rabo da última no meio da troca.
    marcas.append((total + int(0.3 * FS), None))
    x.caixa_marcas = marcas
    return x


def sonda_amostras(longe=False):
    return sonda_layout(longe)[2]


def aplica_caixa(x, fatores, padrao=CAIXA_INICIAL):
    """O que o volume da caixa faria com `x`, para quem não tem caixa (o
    simulador): cada trecho multiplicado pelo fator do passo pedido em
    `x.caixa_marcas`. Marcas além do fim não mudam nada aqui."""
    marcas = getattr(x, 'caixa_marcas', None)
    if not marcas:
        return x
    y = np.array(x, dtype=np.float64)
    pts = sorted((max(0, a), padrao if k is None else k) for a, k in marcas)
    for j, (a, k) in enumerate(pts):
        b = pts[j + 1][0] if j + 1 < len(pts) else len(y)
        y[a:b] *= fatores[k]
    return y


def mede_tons(audio, p1, p2, sessao):
    """Margem de cada tom da grade, em dB, sobre o pior entre o piso em
    silêncio e o que chega naquela frequência enquanto soam tons distantes
    (cauda e intermodulação incluídas -- é contra isso que o 16-FSK decide).
    Mediana de três passos por tom."""
    nominal = LQ + 2 * HUSH + _sonda_tons_nominal()
    r = 1.0 if p2 is None else (p2 - p1) / nominal
    if not 0.995 <= r <= 1.005:
        r = 1.0
    base = p1 + (LQ + HUSH) * r
    a0, w = int(0.008 * FS), int(0.038 * FS)
    n = np.arange(w)
    sondas = np.exp(-2j * np.pi * np.outer(GRADE, n) / FS)

    def energia(ini):
        i = int(round(base + ini * r))
        seg = np.asarray(audio[i:i + w], dtype=np.float64)
        if len(seg) < w:
            return np.zeros(len(GRADE))
        return np.abs(sondas @ seg) ** 2

    sil = [energia(s) for s in range(int(0.02 * FS), int((SONDA_SIL - 0.02) * FS) - w, w)]
    piso = np.median(np.array(sil), axis=0)
    ordem = _ordem_sonda(sessao)
    passo = int(SONDA_PASSO * FS)
    E = np.array([energia(int(SONDA_SIL * FS) + j * passo + a0) for j in range(len(ordem))])
    ordem = np.array(ordem)
    margem = np.zeros(len(GRADE))
    for k in range(len(GRADE)):
        on = np.median(E[ordem == k, k])
        longe = np.abs(ordem - k) >= 3
        interf = np.median(E[longe, k])
        margem[k] = 10 * np.log10(max(on, 1e-30) / max(piso[k], interf, 1e-30))
    return margem


# Acima disto mais margem não muda decisão nenhuma, e perseguir o último dB
# só empurra os tons para fora da banda de sempre.
MARGEM_BASTA = 30.0
CUSTO_SOMBRA = 2.0       # dB por tom pulado
CUSTO_PASSO = 1.0        # dB por passo de grade longe da banda de sempre
PADRAO_16 = 2            # GRADE[2] = 887,5 Hz, o primeiro dos MARY_TONES
PADRAO_PAR = (7, 11)     # 1700/2350, o fsk2


def escolhe_16(margem, sombras_max=SOMBRAS_MAX):
    """Índices na grade dos 16 tons do contrato.

    Faixa deslizante com sombras: uma janela de 16+s tons consecutivos da
    grade, da qual saem os s piores. Vale o pior tom que fica, até
    MARGEM_BASTA; cada sombra custa CUSTO_SOMBRA e cada passo de grade longe
    da banda de sempre custa CUSTO_PASSO. Num canal limpo, portanto, saem os
    tons de sempre sem sombra nenhuma; as sombras só aparecem quando um tom
    está de fato numa cova."""
    m = np.minimum(np.asarray(margem, dtype=float), MARGEM_BASTA)
    melhor = None
    for s in range(sombras_max + 1):
        for a in range(len(GRADE) - 16 - s + 1):
            jan = list(range(a, a + 16 + s))
            fica = sorted(sorted(jan, key=lambda k: m[k])[s:])
            nota = (min(m[k] for k in fica) - CUSTO_SOMBRA * s
                    - CUSTO_PASSO * abs(a - PADRAO_16))
            if melhor is None or nota > melhor[0]:
                melhor = (nota, fica)
    return melhor[1]


def escolhe_par(margem):
    """Dois índices da grade a pelo menos quatro passos (650 Hz), como o fsk2,
    com o mesmo teto de margem e a mesma atração pelo par de sempre."""
    m = np.minimum(np.asarray(margem, dtype=float), MARGEM_BASTA)
    melhor = None
    for a in range(len(GRADE)):
        for b in range(a + 4, len(GRADE)):
            nota = (min(m[a], m[b]) - CUSTO_PASSO * (abs(a - PADRAO_PAR[0])
                                                     + abs(b - PADRAO_PAR[1])))
            if melhor is None or nota > melhor[0]:
                melhor = (nota, (a, b))
    return melhor[1]


@dataclass
class Contrato:
    tons: tuple          # 16 índices na grade
    par: tuple           # 2 índices na grade
    ganho: int           # passo da caixa de quem transmite (0..N_CAIXA-1); o digital é GANHO_FIXO
    degrau: int          # índice em DEGRAUS

    def empacota(self):
        mapa = sum(1 << k for k in self.tons)
        v = mapa | (self.par[0] << 25) | (self.par[1] << 30) | (self.ganho << 35) | (self.degrau << 37)
        return v.to_bytes(5, 'little')

    @classmethod
    def abre(cls, b):
        v = int.from_bytes(bytes(b[:5]), 'little')
        tons = tuple(k for k in range(25) if v >> k & 1)
        if len(tons) != 16:
            raise ValueError("contrato sem 16 tons")
        return cls(tons, ((v >> 25) & 31, (v >> 30) & 31), (v >> 35) & 3, (v >> 37) & 7)

    def descritor(self, degrau=None, nbytes=DADOS_N, modo='auto'):
        d = self.degrau if degrau is None else degrau
        g = GANHO_FIXO
        if modo != 'auto':
            # Degrau é índice em DEGRAUS_MODO[modo]: só a repetição muda. O
            # ganho é o do contrato em todos; os tons são os do contrato no
            # 16-FSK e no 2-FSK e os MFSK_PAIRS fixos no 5x2.
            if modo == '16fsk' and d >= 3:
                return self.descritor(R50 + d - 3, nbytes)
            rep = _REPS_MODO[modo][d]
            nome = f'contrato {DEGRAUS_MODO[modo][d]} {nbytes}B'
            if modo == '16fsk':
                return Descritor(tuple(GRADE[k] for k in self.tons), 4, rep, nbytes, g, nome)
            if modo == '2fsk':
                return Descritor(tuple(GRADE[k] for k in self.par), 1, rep, nbytes, g, nome)
            k = len(MFSK_PAIRS)
            return Descritor(tuple(t for p in MFSK_PAIRS for t in p),
                             k if modo == 'multicanal' else 1, rep, nbytes, g, nome,
                             camada=modo)
        if d == 3:
            return Descritor(tuple(GRADE[k] for k in self.par), 1, 1, nbytes, g,
                             f'contrato 2-FSK {nbytes}B')
        if d in BAUD_GUARDA:
            baud, guarda = BAUD_GUARDA[d]
            return Descritor(tuple(GRADE[k] for k in self.tons), 4, 1, nbytes, g,
                             f'contrato {DEGRAUS[d]} {nbytes}B',
                             baud, guarda, LENTO_PRE, marca='quadro_longo')
        if d == MUITO_LENTO:
            return Descritor(tuple(GRADE[k] for k in self.tons), 4, 2, nbytes, g,
                             f'contrato 16-FSK muito lento {nbytes}B',
                             10, LENTO_GUARDA, LENTO_PRE, marca='quadro_longo')
        if d == LENTO:
            # Marca longa: o lento é o degrau de longe, e lá a de 80 ms
            # fica rente ao limiar.
            return Descritor(tuple(GRADE[k] for k in self.tons), 4, 1, nbytes, g,
                             f'contrato 16-FSK lento {nbytes}B',
                             LENTO_BAUD, LENTO_GUARDA, LENTO_PRE, marca='quadro_longo')
        return Descritor(tuple(GRADE[k] for k in self.tons), 4, (1, 2, 4)[d], nbytes, g,
                         f'contrato {DEGRAUS[d]} {nbytes}B')

    def descreve(self):
        f = [GRADE[k] for k in self.tons]
        falta = [GRADE[k] for k in range(min(self.tons), max(self.tons) + 1) if k not in self.tons]
        return (f"{f[0]:.0f}-{f[-1]:.0f} Hz, sombras {[round(x) for x in falta]}, "
                f"par {GRADE[self.par[0]]:.0f}/{GRADE[self.par[1]]:.0f}, "
                f"caixa {self.ganho}, ganho {GANHO_FIXO}, {DEGRAUS[self.degrau]}")


PICO_SATURA = 0.98           # pico de entrada a partir do qual o conversor do microfone cortou


def margem_llr(llr, ref):
    """Margem de um bloco conhecido, em dB: o 10º percentil do
    log-likelihood de cada bit com o sinal do bit certo -- a folga dos bits
    mais fracos, que são os que derrubam um bloco. O llr do 16-FSK é
    max-log, a razão de energia entre o melhor tom com o bit em 1 e o melhor
    com ele em 0 (`MaryDemodulator.demodulate_soft`), em log natural; 10/ln 10
    o põe em dB. Ao contrário do erro de bit, não satura em 0%. Negativa quando
    mais de 10% dos bits saíram errados."""
    sinal = np.where(np.asarray(ref) > 0, 1.0, -1.0)
    return float(np.percentile(np.asarray(llr) * sinal, 10)) * 10 / np.log(10)


# Subir a caixa um passo só vale se a margem subir pelo menos isto; abaixo,
# é o joelho -- a cadeia parou de responder ao volume, ou começou a
# comprimir -- e a escolha para ali.
MARGEM_SOBE_MIN = 1.0
# Microfone: o de 0,6 só vence o de partida com isto a mais de margem.
MARGEM_MIC_MIN = 1.0


def sobe_caixa(celulas):
    """Escolha da caixa por subida: `celulas` [(passo, ber, inteiro, margem
    ou None, pico ou None)] de um mesmo microfone, em ordem de passo. Começa
    no passo mais baixo que decodifica (erro de bit abaixo de
    LIMIARES_BER[2], sem corte do conversor) e sobe enquanto o próximo passo
    também decodifica e ganha MARGEM_SOBE_MIN dB; fica com o de maior margem
    até onde parou. Devolve (índice em `celulas` ou None, motivo)."""
    def bom(c):
        return (c[3] is not None and c[1] < LIMIARES_BER[2]
                and (c[4] is None or c[4] < PICO_SATURA))
    ini = next((i for i, c in enumerate(celulas) if bom(c)), None)
    if ini is None:
        return None, "nenhum passo decodificou sem corte"
    i = ini
    while True:
        if i + 1 >= len(celulas):
            motivo = "subiu até o último passo da escada, a margem ainda crescendo"
            break
        c, n = celulas[i], celulas[i + 1]
        if not bom(n):
            motivo = (f"parei no passo {c[0]}: o {n[0]} "
                      f"{'cortou no conversor' if n[4] is not None and n[4] >= PICO_SATURA else 'não decodificou'}")
            break
        if n[3] < c[3] + MARGEM_SOBE_MIN:
            d = n[3] - c[3]
            motivo = (f"parei no passo {c[0]}: no {n[0]} a margem "
                      f"{'caiu' if d < 0 else 'subiu só'} {abs(d):.1f} dB")
            break
        i += 1
    melhor = max(range(ini, i + 1), key=lambda j: (celulas[j][3], -j))
    return melhor, motivo


def escolhe_celula(sondas_g, caixas, mics, picos=None):
    """(índice na escada, passo de caixa, passo de microfone) da sonda que
    vence. Vale o acerto de bits; dentro de um ponto do melhor, o menor
    passo de caixa -- uma caixa mais baixa que acerta tanto quanto a mais
    alta é a cadeia saindo da compressão, e a folga sobra para o dia em que
    a sala mudar --, e no empate o microfone de partida. Uma sonda que chegou
    com o conversor cortando (pico >= PICO_SATURA) só vence se todas
    cortaram: acertar hoje no corte é acertar por sorte."""
    acerto = [1 - b for b, _ in sondas_g]
    validas = [j for j in range(len(acerto))
               if picos is None or picos[j] is None or picos[j] < PICO_SATURA]
    if not validas:
        validas = list(range(len(acerto)))
    top = max(acerto[j] for j in validas)
    if top < 1 - LIMIARES_BER[-1]:
        # Nenhuma sonda lida (todas a 0,5 quando nem a marca foi achada): o
        # "menor passo que acerta tanto quanto o melhor" seria o mais baixo,
        # o oposto do que um canal assim pede. De longe é o caso comum.
        j = max(range(len(acerto)), key=lambda j: (caixas[j], -mics[j]))
        return j, caixas[j], 0
    j = min((j for j in validas if acerto[j] >= top - 0.01),
            key=lambda j: (caixas[j], mics[j]))
    return j, caixas[j], mics[j]


def degrau_de(ber, inteiro):
    if ber < LIMIARES_BER[0] and inteiro:
        return 0
    if ber < LIMIARES_BER[1]:
        return 1
    if ber < LIMIARES_BER[2]:
        return 2
    if ber < LIMIARES_BER[3]:
        return 3
    if ber < LIMIARES_BER[4]:
        # Nem o 16-FSK a 100 baud foi lido com menos de 35% de erro: o canal
        # não segura símbolo de 10 ms. Começa no lento.
        return LENTO
    return MUITO_LENTO


def _seg_dados(d):
    """Quanto um quadro de dados leva no ar no degrau `d` (a taxa útil, ao
    contrário: menos é mais rápido). Tons não mudam a duração."""
    return Contrato(tuple(range(16)), (0, 1), 0, d).descritor(d, DADOS_N).segundos


def escolhe_velocidade(vres):
    """O degrau pela bateria de velocidade. `vres`: (ber, inteiro, margem
    ou None) por bloco, na ordem de VELOCIDADES. Aceito: nos 100 baud, o
    degrau que o erro de bit permite até o rep4 (`degrau_de`, o rep1 só
    inteiro); nos mais lentos, o bloco inteiro com erro abaixo de
    LIMIARES_BER[1] -- é rep1 como os dados naquele degrau. Vence o que
    passa mais rápido um quadro de dados; um mais lento o troca se custar
    menos de VEL_GANHO_MIN e tiver VEL_MARGEM_MELHOR dB a mais.
    Devolve (degrau ou None se nada passou, texto para o diário)."""
    # Subindo do mais lento: o primeiro degrau que falha encerra a subida, e o
    # contrato fica no último que passou -- um 100 baud lido depois de um 50
    # que falhou não conta.
    aceitos, partes, parou = [], [], False
    for (b, _g, _p, _m, deg), (ber, inteiro, marg) in zip(VELOCIDADES, vres):
        if parou:
            partes.append(f"{b} baud não conta (a subida parou antes)")
            continue
        mtxt = '' if marg is None else f", margem {marg:.1f} dB"
        if marg is None:
            partes.append(f"{b} baud não lido -- não ouvi; a subida para aqui")
            parou = True
            continue
        if deg == 0:
            d = degrau_de(ber, inteiro)
            ok = d <= 2
        else:
            d, ok = deg, inteiro and ber < LIMIARES_BER[1]
        partes.append(f"{b} baud {'ok' if ok else 'falhou'} ({ber * 100:.0f}%"
                      f"{', inteiro' if inteiro else ''}{mtxt})")
        if ok:
            aceitos.append((_seg_dados(d), marg, d, b))
        else:
            parou = True
    if not aceitos:
        return None, f"velocidade: {'; '.join(partes)}; nenhum passou"
    t, m, d, b = min(aceitos)
    porque = f"o mais rápido aceito ({t:.1f} s por quadro de {DADOS_N} B)"
    troca = [a for a in aceitos if a[0] <= t * VEL_GANHO_MIN and a[1] >= m + VEL_MARGEM_MELHOR
             and a[2] != d]
    if troca:
        t2, m2, d2, b2 = max(troca, key=lambda a: a[1])
        porque = (f"{DEGRAUS[d2]} custa só {(t2 / t - 1) * 100:.0f}% a mais que {DEGRAUS[d]} "
                  f"e tem {m2 - m:.1f} dB a mais de margem")
        d = d2
    return d, f"velocidade: {'; '.join(partes)}; escolho {DEGRAUS[d]}: {porque}"


def decide(margem, sondas_g, caixas=None, mics=None, picos=None):
    """O contrato de um sentido, a partir do que o receptor mediu, e o passo
    de microfone que ele escolheu para si.

    `sondas_g`: (ber, inteiro) por sonda da escada, ber 0,5 quando a sonda
    nem foi achada; `caixas`/`mics`, o passo de cada uma."""
    caixas = tuple(range(len(sondas_g))) if caixas is None else caixas
    mics = (0,) * len(sondas_g) if mics is None else mics
    j, k, mic = escolhe_celula(sondas_g, caixas, mics, picos)
    ber, inteiro = sondas_g[j]
    return Contrato(tuple(escolhe_16(margem)), escolhe_par(margem), k,
                    degrau_de(ber, inteiro)), mic, j


# --- Ouvido: varredura contínua do que chega ---------------------------------

class Ouvido:
    """Guarda o áudio que chega e acha as marcas nele, continuamente.

    Nunca "toca e depois grava uma janela": uma chamada que caísse na borda
    de uma janela seria cortada ao meio e nunca achada, o que pareceria uma
    sala ruim. O áudio entra em pedaços de qualquer tamanho e a varredura
    anda sobre ele com sobreposição. O que a própria estação tocou (mais uma
    guarda) é zerado ao entrar e excluído da busca: o microfone ouve o
    próprio alto-falante muito mais alto que o outro lado.
    """

    PASSO = FS // 2
    GUARDA_MAX = 90 * FS

    def __init__(self):
        self.buf = np.zeros(0, dtype=np.float32)
        self.base = 0
        self.fim = 0
        self.cegos = []
        self.varrido = 0
        self.marcas = []          # (pos, tipo, score), pos = início da marca
        self.fracas = []          # idem, de quadro, entre LIMIAR_DIRIGIDO e o limiar
        self.lmax = max(len(m) for m in MOLDES.values())

    def cego(self, a, b):
        self.cegos.append((a, b))
        self.cegos = [(x, y) for x, y in self.cegos if y > self.base]

    def _invalido(self, a, b):
        inv = np.zeros(b - a, dtype=bool)
        if a < self.base:
            inv[:self.base - a] = True
        for x, y in self.cegos:
            lo, hi = max(x, a), min(y, b)
            if lo < hi:
                inv[lo - a:hi - a] = True
        return inv

    def acrescenta(self, x):
        x = np.asarray(x, dtype=np.float32).copy()
        a = self.fim
        x[self._invalido(a, a + len(x))] = 0.0
        self.buf = np.concatenate([self.buf, x])
        self.fim += len(x)
        if len(self.buf) > self.GUARDA_MAX:
            corta = len(self.buf) - self.GUARDA_MAX
            self.buf = self.buf[corta:]
            self.base += corta

    def trecho(self, a, b):
        a2, b2 = max(a, self.base), min(b, self.fim)
        out = np.zeros(max(0, b - a), dtype=np.float64)
        if a2 < b2:
            out[a2 - a:b2 - a] = self.buf[a2 - self.base:b2 - self.base]
        return out

    def varre(self, fixas=frozenset()):
        """Avança a varredura até onde o áudio permite; devolve as marcas novas.
        `fixas`: posições de marcas já usadas, que nenhuma outra substitui."""
        novas = []
        L = self.lmax
        while self.fim - self.varrido >= self.PASSO + 2 * L:
            r0, r1 = self.varrido, self.varrido + self.PASSO
            a = r0 - L
            x = sosfiltfilt(_BANDA, self.trecho(a, r1 + 2 * L))
            inv = np.concatenate([[0], np.cumsum(self._invalido(a, r1 + 2 * L))])
            achadas = []
            e = np.concatenate([[0.0], np.cumsum(x * x)])
            for tipo, m in MOLDES.items():
                lt = len(m)
                c = np.abs(fftconvolve(x, m[::-1], mode='valid'))
                q = np.arange(len(c))
                ruim = (inv[q + lt] - inv[q]) > 0
                en = np.sqrt(np.maximum(e[q + lt] - e[q], 0.0)) * np.linalg.norm(m)
                s = np.where(ruim | (en <= 0), 0.0, c / np.maximum(en, 1e-12))
                h = lt // 2
                if tipo in QUADROS:
                    for i in np.flatnonzero((s >= LIMIAR_DIRIGIDO) & (s < LIMIARES[tipo])):
                        pos = a + i
                        if r0 <= pos < r1 and s[i] >= s[max(0, i - h):i + h + 1].max():
                            self.fracas.append((pos, tipo, float(s[i])))
                for i in np.flatnonzero(s >= LIMIARES[tipo]):
                    pos = a + i
                    if not r0 <= pos < r1:
                        continue
                    if s[i] < s[max(0, i - h):i + h + 1].max():
                        continue
                    achadas.append((pos, tipo, float(s[i])))
            # Uma varredura também correlaciona, mais fraco, com o molde da
            # outra família. No mesmo lugar, fica a mais forte.
            achadas.sort(key=lambda t: -t[2])
            for pos, tipo, sc in achadas:
                # A curta e a longa de quadro são a mesma varredura em 80 e
                # 400 ms: uma correlaciona com a outra. As duas ficam; quem
                # escolhe é a estação, pelo que o estado dela espera.
                rivais = [m for m in novas + self.marcas[-8:]
                          if m[0] >= r0 - L and abs(pos - m[0]) < int(0.15 * FS)
                          and {m[1], tipo} != set(QUADROS)]
                # Uma marca fraca guardada no pedaço anterior não pode calar
                # uma forte que só aparece neste. A forte substitui a fraca --
                # só se a fraca for de quadro ou sonda (a chamada e a resposta
                # já agiram em `_marca`) e ainda não tiver sido usada por um
                # segmento.
                fracas = [m for m in rivais if m in self.marcas and m[2] < sc
                          and m[1] in QUADROS + ('sonda',) and m[0] not in fixas]
                if len(fracas) != len(rivais):
                    continue
                for m in fracas:
                    self.marcas.remove(m)
                novas.append((pos, tipo, sc))
            self.varrido = r1
        self.fracas = [m for m in self.fracas if m[0] >= self.base]
        novas.sort()
        self.marcas += novas
        return novas


# --- Estação ------------------------------------------------------------------

RESP = int(0.8 * FS)           # pausa antes de responder
# A vez de cada um. Só o chamador abre um turno; o ouvinte responde numa vez
# que as duas pontas calculam igual, contada do fim do turno do chamador:
# começa RESP_VEZ depois (mais o que leva ler um turno longo, e mais de
# longe), e se o ouvinte não estiver pronto até TOL_VEZ depois disso, cala e
# espera o próximo turno do chamador. O chamador, que sabe quando a vez do
# ouvinte começa e quanto ela pode durar, nunca fala dentro dela -- nem para
# repetir.
RESP_VEZ = int(2.5 * FS)
TOL_VEZ = int(1.5 * FS)
LAT_OUTRO = int(1.5 * FS)      # latência de saída do outro lado, que daqui não se mede
FIM_TENTATIVAS = 3


def atraso_vez(n, esforco):
    """Do fim de um turno de `n` amostras do chamador ao começo da vez do
    ouvinte. Cresce com o turno porque a leitura cresce com ele."""
    return RESP_VEZ + n // 8 + 4 * FS * esforco


# Quanto o ouvinte espera o OLA depois de responder a uma chamada: a escada
# inteira do chamador, cada degrau duas vezes (a segunda com a caixa mais
# alta), cada vez o OLA, o OLA_OK, a vez e as latências. ~4 min.
# Acertar o som em duas escadas, só com varreduras.
# O chamador sobe devagar: cada passo k da caixa é uma JANELA -- uma chamada
# e o resto da janela ouvindo. Quem ouve a chamada bem responde numa escada
# RÁPIDA dentro dela: uma resposta curta a cada `_passo_resposta`, da caixa
# mais baixa para cima. O chamador, ao ouvir uma resposta, confirma na hora
# (duas varreduras de chamada) e trava em k -- ouvido em k e não em k-1,
# porque quem ouve responde sempre. O ouvinte, pela hora em que a
# confirmação chega, sabe qual resposta foi ouvida e trava nela. Todos os
# tempos saem das latências medidas no próprio eco e de:
VARRE_ATRASO = int(0.9 * FS)   # a varredura de marcas anda até ~0,9 s atrás do presente
LAT_SOM_OUTRO = int(0.8 * FS)  # latência de saída do outro lado, que daqui não se mede
RESP_LEAD = int(0.3 * FS)      # silêncio antes de cada resposta: a caixa troca de passo nele
CONFIRMA_GAP = int(0.1 * FS)
MARGEM_SOM = int(0.5 * FS)
# Refino pelo som, depois de travar: quem toca sobe um passo e chama (ou
# responde) de novo; quem ouve compara com a anterior e diz "melhorou" ou "não
# melhorou" (cada um com a sua marca). Primeiro a caixa do chamador, depois a do
# ouvinte. Melhorou = correlação normalizada acima da anterior por mais que
# REF_CORR (o que a sala varia sozinha); empatada (dentro de REF_CORR), a
# relação sinal/ruído sobre o piso acima por REF_SNR (metade de um passo de
# caixa, ~4 dB).
REF_CORR = 0.03
REF_SNR = 2.0
ASSENTA = int(0.8 * FS)        # no refino, quanto ouvir depois da 1ª resposta antes de decidir
DUPLA_ATE = int(0.75 * FS)     # depois de uma chamada, quanto esperar a 2ª varredura de uma dupla

OLA_ESPERA_TOTAL = sum(
    2 * (2 * ABORDAGENS[i].amostras + atraso_vez(ABORDAGENS[i].amostras, int(j >= OLA_LONGE_DESDE))
         + TOL_VEZ + 2 * LAT_OUTRO + 3 * FS)
    for j, i in enumerate(OLA_ESCADA))
GUARDA = int(0.5 * FS)         # cegueira depois de tocar
ESPERA = int(3.0 * FS)         # quanto esperar o começo de uma resposta
# Quanto esperar outro segmento num turno. Tem de cobrir o atraso da
# varredura (PASSO + 2 moldes), senão o segmento seguinte -- que começa 0,1 s
# depois -- ainda não foi achado quando a folga vence.
FOLGA = Ouvido.PASSO + 2 * max(len(m) for m in MOLDES.values()) + int(0.5 * FS)
import os as _os
_DEPURA = bool(_os.environ.get('ENLACE_DEPURA'))
# Com o canal aberto, uma chamada acima do limiar dela pode derrubar a sessão:
# o falso de uma varredura casando com um quadro de dados cai antes, em
# "chamada dentro de outro som", e o silêncio exigido abaixo (SILENCIO_MORTO
# sem nada legível) afasta o resto.
LIMIAR_RECOMECO = LIMIARES['chamada']
CONVERSA_CONSULTA = 15 * FS    # conversa: chamador sem nada a mandar consulta o ouvinte a cada tanto
SILENCIO_MORTO = 90 * FS       # canal aberto sem nada legível: dado por morto (o muito lento leva 52 s por quadro de dados)


class Estacao:
    """Uma ponta do enlace. `papel` é 'chamador' ou 'ouvinte'.

    Uso: a cada pedaço de áudio, `ouvir(pedaço)` e depois `passo()`; se
    `passo` devolver amostras, tocá-las já. O relógio é o número de amostras
    ouvidas, e o que for tocado é tido como começando no instante atual.
    """

    def __init__(self, papel, semente=None, log=None, nome=None, modo_dados='auto'):
        assert papel in ('chamador', 'ouvinte')
        if modo_dados not in MODOS_DADOS:
            raise ValueError(f"modo de dados {modo_dados!r}; um de {MODOS_DADOS}")
        self.papel = papel
        self.modo_dados = modo_dados
        self.degraus = DEGRAUS_MODO[modo_dados]
        self.ultimo_degrau = len(self.degraus) - 1
        self.nome = nome or papel
        self.rng = np.random.default_rng(semente)
        self.ouvido = Ouvido()
        self._log = log or (lambda s: None)
        self.eventos = []          # (t, texto)
        self.recebido = bytearray()
        self.fila = bytearray()
        self.pendente_tx = None
        self.saudado = False
        self.aberto = False
        self.contrato_tx = None    # o contrato do sentido em que eu transmito
        self.contrato_rx = None
        self.ctrl = None
        self.sessao = 0
        self.agenda = []           # [(instante, amostras)]
        self.tocando = []          # rótulos do último turno agendado, para o painel
        self._no_ar = None         # (início, amostras, rótulo) do que saiu por último
        self.pendente = None       # segmento em leitura: (pos, tipo, prazo)
        self.turno = []            # eventos do turno que está chegando
        self.turno_fim = None      # instante em que o turno se dá por encerrado
        self.ouve_ate = 0          # o turno do outro lado não acaba antes disto
        self._terminal = False     # o último segmento lido fecha o turno
        self.outro_tem_mais = False  # o outro lado mandou DADOS: pode haver mais
        self.sondar_em = None      # chamador ocioso: próxima consulta ao outro lado
        self.conversa = False      # canal fica aberto: sem FIM, consulta o ouvinte de tempos em tempos
        self.turno_ultimo = 0      # fim do último segmento lido
        self._cursor = 0           # marcas antes disto já foram tratadas
        self._eco_janela = (0, 0)  # quando tocou a minha última varredura de saudação
        self.ultimo_turno = None   # o que eu toquei por último, para repetir
        self.espera_ate = None     # prazo para o começo de uma resposta
        self.tentativas = 0
        self.ab = 0                # abordagem de controle em uso
        self.esforco = 0           # 1 = longe: o OLA saiu a 20 baud ou menos (decide o chamador)
        self._chamou_fim = None    # fim da minha última chamada; resposta antes disso é falsa
        self.k_chamada = 0
        self.livre_em = 0          # fim do que estou tocando
        self.ouvido_em = -10 * FS  # última marca ouvida
        self._chamada_em = None    # a última chamada que respondi
        self._resp_len = 0         # duração prevista da resposta ao meu último turno
        self._usadas = set()       # marcas já lidas como parte de um segmento
        self.lat_extra = 0         # latência de saída não declarada, medida no próprio eco
        self._eco = None           # (tipo, início) da última varredura de saudação tocada
        self._niveis = []          # (instante, rms) de 0,25 s fora da cegueira, para o piso
        self._turno_ini = 0        # abertura do turno que está chegando
        self._lido_ini = 0         # abertura do último segmento lido
        self.encerrada = False     # despediu-se: tudo entregue nos dois sentidos
        self.fim_tentativas = 0    # FIMs mandados sem resposta
        self._piso = None
        self._nivel_em = 0
        self.mic = 0               # passo de microfone em uso (0: o de partida)
        self.pedidos_mic = []      # [(instante, passo)]: o runtime troca o microfone quando chegar
        self.pedidos_mic_passo = []  # [(instante, +1/-1, pico)]: o runtime move o microfone um passo
        self.mic_no_teto = False   # o runtime avisa que o microfone não sobe mais
        self.mic_no_piso = False   # ... nem desce
        self._ouvi_chamada = None  # (pico, rms) da última chamada que respondi
        self._ouvi_resp = None     # (pico, rms) da resposta que me fez mandar OLA
        self._nivel_quadro = None  # (pico, rms) do último quadro lido
        self._disse_indep = False
        self._resp_na_janela = 0   # respostas ouvidas na janela de chamada em curso (chamador)
        self._resp_tocadas = []    # [(instante em que soou, passo)] da minha escada rápida (ouvinte)
        self._resp_fim = 0
        self._plano_mic = None     # marca de abertura da sondagem em que o microfone varre
        self.versao_outro = None
        self.pedidos_caixa = []    # [(instante, passo absoluto)]: o runtime põe a caixa já
        self.reinicia()
        self.anota("tempos no ar: " + "; ".join(
            f"{d.nome} {d.segundos:.1f}s" for d in ABORDAGENS)
            + f"; sondagem {sonda_amostras() / FS:.1f}s")
        self.anota(f"modo de dados: {modo_dados} (degraus: {', '.join(self.degraus)}); "
                   "a outra ponta precisa do mesmo")

    # --- utilidades

    @property
    def agora(self):
        return self.ouvido.fim

    def anota(self, s):
        t = self.agora / FS
        self.eventos.append((t, s))
        self._log(f"[{self.nome} {t:7.2f}s] {s}")

    def enviar(self, dados):
        self.fila += bytes(dados)

    def reinicia(self, mantem_som=False):
        """Volta à saudação. `mantem_som`: a escada do OLA inteira falhou, ou
        o ouvinte cansou de esperar o OLA -- a caixa e o microfone ficam onde
        estão (o som já foi subido para isso), e a chamada recomeça dali."""
        k_atual = self.caixa() if mantem_som else None
        resp_passo = getattr(self, '_resp_passo', None) if mantem_som else None
        self.estado = 'chamando' if self.papel == 'chamador' else 'escutando'
        self.proxima_chamada = self.agora + int(self.rng.uniform(0.2, 1.0) * FS)
        self.saudado = self.aberto = False
        self.contrato_tx = self.contrato_rx = self.ctrl = None
        self.ultimo_turno = None
        self.espera_ate = None
        self.tentativas = 0
        self.sem_progresso = 0
        self.encerrada = False
        self.fim_tentativas = 0
        # Acertar o som recomeça do passo mais baixo: `trava` é o passo
        # absoluto em que o outro lado me ouviu, None enquanto não se sabe.
        self.trava = None
        self.k_som = 0             # índice em CHAMADA_PASSOS: a minha chamada, ou a minha resposta
        self._resp_passo = resp_passo  # o passo em que toquei a minha última resposta
        self._resp_baixas = 0      # respostas que ouvi (o chamador), boas ou baixas
        self.k_ola = 0             # índice em OLA_ESCADA
        self._ola_subiu = False    # neste degrau do OLA a caixa já subiu uma vez
        self._ola_lido = None      # (sessão, abordagem) do último OLA a que respondi
        self._disse_indep = False
        self._ouvi_resp = None
        self._resp_na_janela = 0
        self._resp_tocadas = []
        self._dupla = None         # (pos, sc) de uma chamada que pode ser a 1ª de uma dupla
        self._ref = None           # o refino em curso
        self._q_ant = None         # (correlação, SNR, rms) da última varredura com que comparo
        if mantem_som:
            self.k_som = next((i for i, p in enumerate(CHAMADA_PASSOS) if p >= k_atual),
                              len(CHAMADA_PASSOS) - 1)
            self._pede_caixa(CHAMADA_PASSOS[self.k_som])
            return
        self._pede_caixa(CHAMADA_PASSOS[0])
        if getattr(self, 'mic', 0):
            self.mic = 0
            self.pedidos_mic.append((self.agora, 0))

    def caixa(self):
        """O passo absoluto da minha caixa fora da escada: o do contrato que
        recebi para o que envio, relativo à trava; antes dele, a trava; antes
        da trava, o passo da escada de acertar o som em que estou."""
        if self.contrato_tx is not None:
            return self._abs(self.contrato_tx.ganho)
        if self.trava is not None:
            return self.trava
        return CHAMADA_PASSOS[min(self.k_som, len(CHAMADA_PASSOS) - 1)]

    def _abs(self, k):
        """Passo relativo da escada da sondagem/contrato (0..N_CAIXA-1) para
        o absoluto: em volta da trava, limitado à tabela."""
        base = self.trava if self.trava is not None else CAIXA_INICIAL
        return int(min(N_POS - 1, max(0, base + CAIXA_REL[k])))

    def _pede_caixa(self, k):
        """Põe a caixa no passo absoluto `k` já, antes do próximo som: a troca no
        início do próprio som pode chegar atrasada, e a varredura da chamada tem
        0,2 s."""
        self.pedidos_caixa.append((self.agora, k))

    def _trava(self, k, porque):
        if self.trava == k:
            return
        self.trava = k
        self._pede_caixa(k)
        self.anota(f"[volume] som acertado: caixa no passo {k} ({CAIXA_POS[k]:.0%} nominal); {porque}")

    def _nivel(self, pos, tipo):
        """(rms, pico) da varredura `tipo` que começa em `pos`."""
        L = len(MOLDES[tipo])
        r = self._rms_valido(pos, pos + L)
        x = self.ouvido.trecho(pos, pos + L)
        return r, (float(np.max(np.abs(x))) if len(x) else 0.0)

    def _toca(self, partes, depois=0, espera=True, guarda_turno=True, apos=None, eco=None):
        """Agenda um turno: segmentos separados por GAP, `depois` amostras
        após o fim do que chegou por último (ou a partir de `apos`)."""
        if self.encerrada:
            return
        s = []
        padrao = self.caixa()
        marcas, o = [(0, padrao)], 0
        for p in partes:
            for a, k in getattr(p, 'caixa_marcas', None) or ():
                marcas.append((o + a, padrao if k is None else self._abs(k)))
            o += len(p) + GAP
            s += [p, np.zeros(GAP)]
        x = np.concatenate(s[:-1]).view(Parte)
        x.caixa_marcas = marcas
        quando = max(self.agora, (self.turno_ultimo if apos is None else apos) + depois)
        # Um turno novo substitui o que ainda espera a vez, nunca se soma a
        # ele: a decisão mais recente é a que vale, e dois turnos na fila
        # sairiam colados, o segundo por cima da resposta do outro lado. O que
        # já está tocando saiu da agenda em `passo` e não é tocado aqui.
        self.agenda = [(quando, x, eco)]
        self.tocando = [getattr(p, 'rotulo', None) or f"{len(p) / FS:.1f}s" for p in partes]
        self.anota(f"toco em {quando / FS:.1f}s: " + " + ".join(self.tocando))
        if guarda_turno:
            self.ultimo_turno = partes
        # A resposta só começa depois que o outro lado varreu e decodificou o
        # que eu disse, e isso cresce com o tamanho do turno, e mais de longe,
        # onde a varredura com a marca longa anda mais atrás. E a resposta
        # inteira, não só o começo dela: se a marca de abertura se perder, é o
        # comprimento previsto que impede falar por cima.
        self._resp_len = self._vez_do_outro()
        self.espera_ate = self._fim_da_vez(quando, len(x)) if espera else None

    def _fim_da_vez(self, a, n):
        """Até quando a resposta a um turno de `n` amostras tocado em `a` pode
        chegar: a vez do outro lado inteira, com a tolerância dele, a
        latência das duas saídas e a resposta mais longa que o estado admite.
        Antes disto, repetir seria falar por cima de uma resposta."""
        return (a + self.lat_extra + n + atraso_vez(n, self.esforco) + TOL_VEZ
                + LAT_OUTRO + self._resp_len + 2 * Ouvido.PASSO)

    def _minha_vez(self, partes, espera=True):
        """O ouvinte fala só na vez dele: a partir de RESP_VEZ (mais a leitura)
        depois do fim do turno do chamador, e nunca mais tarde que TOL_VEZ
        disso. Atrasado, cala -- o chamador repete o turno, e a vez volta."""
        if not self._terminal and not (self.aberto and self.confirmado):
            # O fim do turno do chamador só se sabe ao certo pelo último
            # segmento dele, lido: um ilegível no fim deixa o instante em
            # aberto, e uma vez contada de um fim errado cairia em cima da
            # repetição dele. Calo; ele repete, e desce o degrau se preciso.
            self.anota("turno do chamador não lido até o fim; calo e espero a repetição")
            return False
        if not self._terminal:
            # Com o canal aberto, calar custaria a vez inteira do chamador em
            # silêncio dos dois lados até o prazo dele vencer. O turno só chega
            # aqui com o som já de volta ao piso (`passo` espera `_ocupado`),
            # então respondo logo, e o ACK com o último seq que li é o "não li,
            # repete" -- ele reenvia o pendente na hora. Mas o silêncio não prova
            # que ele parou: o 16-FSK lento chega picotado, com buracos onde os
            # tons caem nas covas da sala. O fim seguro é o do quadro mais longo
            # que ele pode estar tocando, contado da abertura do turno; e o "não
            # li" é só o ACK, curto, nunca os meus DADOS.
            n = self.contrato_rx.descritor(self._degrau(), DADOS_N, self.modo_dados).amostras
            fim = max(self.turno_ultimo, self._turno_ini + n)
            quando = max(self.agora + int(0.3 * FS), fim + atraso_vez(n, self.esforco))
            d = self.contrato_tx.descritor(self._degrau(), CTRL_N, self.modo_dados)
            ack = self._quadro(d, ACK, bytes([self.ultimo_rx, self._degrau()]))
            self.anota(f"turno do chamador não lido até o fim; respondo 'não li' "
                       f"(ACK do seq {self.ultimo_rx}) em {(quando - self.agora) / FS:.0f} s, "
                       "depois do fim mais tardio possível do turno dele")
            self._toca([ack], apos=quando, espera=espera)
            return True
        # O tamanho que conta para a leitura é o do último segmento lido, não
        # desde a primeira marca do turno: uma marca falsa (eco) antes dele
        # inflaria a conta, e a vez sairia depois do prazo do chamador. Menor
        # nunca é perigoso: a vez continua começando depois do fim de verdade
        # do turno.
        n = max(0, self.turno_ultimo - max(self._turno_ini, self._lido_ini))
        quando = self.turno_ultimo + atraso_vez(n, self.esforco)
        if self.agora > quando + TOL_VEZ:
            self.anota(f"perdi a minha vez ({(self.agora - quando) / FS:.1f}s tarde); "
                       "calo e espero o chamador repetir")
            return False
        self._toca(partes, apos=quando, espera=espera)
        return True

    def _quadro(self, desc, tipo, carga, mais=False):
        x = modula_quadro(desc, empacota(tipo, self.sessao, carga, desc.nbytes, mais)).view(Parte)
        x.rotulo = (f"{NOMES.get(tipo, tipo)}{f' seq {carga[0]}' if tipo == DADOS else ''} em "
                    f"{desc.nome or 'controle'}, {desc.baud} baud, ganho {desc.ganho}, "
                    f"{len(x) / FS:.1f}s")
        return x

    def _ganho_chamada(self):
        g = GANHOS_CHAMADA[self.esforco]
        self.k_chamada += 1
        return g[(self.k_chamada - 1) % len(g)]

    def _poe_esforco(self, e, porque):
        """Longe (1) ou perto (0). Quem decide é o chamador, pelo degrau do
        OLA, e o OLA leva a decisão; o ouvinte a adota ao lê-lo."""
        if e != self.esforco:
            self.esforco = e
            self.anota(f"{porque}: " + ("longe -- marca longa, sondagem a 20 baud" if e
                                        else "perto"))

    def _ab_da_escada(self, porque):
        """Põe a abordagem do OLA no degrau `k_ola` da escada, e o esforço
        com ela: a 20 baud ou menos é longe."""
        self.ab = OLA_ESCADA[self.k_ola]
        self._poe_esforco(int(self.k_ola >= OLA_LONGE_DESDE), porque)

    def _quero(self):
        """A marca de quadro que o estado espera, ou None se qualquer uma.
        O ouvinte antes de ler o OLA não sabe em que degrau ele vem: qualquer uma."""
        if self.estado == 'ola':
            return ABORDAGENS[self.ab].marca
        if self.papel == 'ouvinte' and not self.saudado:
            return None
        if self.aberto and self.contrato_rx is not None:
            return self.contrato_rx.descritor(self._degrau(), DADOS_N, self.modo_dados).marca
        if self.ctrl is not None:
            return self.ctrl.marca
        return 'quadro_longo' if self.esforco else 'quadro'

    def _sonda(self):
        x = modula_sonda(self.sessao, SONDA_GANHO_LONGE if self.esforco else SONDA_GANHO_TOM,
                         longe=bool(self.esforco))
        x.rotulo = (f"SONDA (tons em degraus, escada de volume da caixa"
                    f"{' a 20 baud' if self.esforco else ' duas vezes, a 2ª com o microfone de lá 8 dB abaixo'}, depois a velocidade a 20, 33, 50 e 100 baud), "
                    f"{len(x) / FS:.1f}s")
        return x

    def _degrau(self):
        """O degrau em que falo e o que peço: um só para os dois sentidos, o
        mais lento dos dois. Cada sentido mede o seu canal (tons, par, ganho
        continuam por sentido), mas a velocidade é do enlace: as duas pontas
        falam igual, e quem desce puxa a outra junto."""
        return max(self.degrau_tx, self.degrau_rx)

    # --- laço

    def desloca_saida(self, n):
        """O que `passo` acabou de devolver só começa a soar `n` amostras
        depois do instante atual da estação: áudio de entrada ainda na fila
        do runtime, mais a latência da saída (num alto-falante Bluetooth, uma
        fração apreciável de segundo). Sem isto a cegueira cobriria o
        intervalo errado e a estação se ouviria."""
        if n <= 0:
            return
        if self._eco is not None:
            self._eco = (self._eco[0], self._eco[1] + n)
        if self._eco_janela[1] > self.agora:
            self._eco_janela = (self._eco_janela[0] + n, self._eco_janela[1] + n)
        a, b = self.ouvido.cegos[-1]
        self.ouvido.cegos[-1] = (a + n, b + n)
        self.livre_em += n
        if self.espera_ate is not None:
            self.espera_ate += n
        if self._chamou_fim is not None and self._chamou_fim > self.agora:
            self._chamou_fim += n          # a chamada que acabou de sair

    def ouvir(self, x):
        self.ouvido.acrescenta(x)

    def _mede_piso(self):
        """Piso de ruído: percentil 10 do rms de 0,25 s dos últimos 120 s
        (mais longo que um quadro muito lento, que tem 52 s)."""
        if self.agora - self._nivel_em < FS // 4:
            return
        self._nivel_em = self.agora
        r = self._rms_valido(self.agora - FS // 4, self.agora)
        if r is None:
            return
        self._niveis = [(t, v) for t, v in self._niveis if t > self.agora - 120 * FS]
        self._niveis.append((self.agora, r))
        if len(self._niveis) >= 8:
            self._piso = float(np.percentile([v for _, v in self._niveis], 10))

    def _ocupado(self):
        """Escutar antes de falar: o outro lado ainda está tocando? Não diz
        o quê nem se dá para ler -- só que há som acima do piso. Longe o sinal
        chega só poucos dB acima do piso, daí 1,5x."""
        if not self._piso:
            return False
        r = self._rms_valido(self.agora - FS // 2, self.agora)
        return r is not None and r > 1.5 * self._piso

    def passo(self):
        self._mede_piso()
        fixas = set(self._usadas)
        if self.pendente is not None:
            fixas.add(self.pendente['pos'])
        for pos, tipo, sc in self.ouvido.varre(fixas):
            self._marca(pos, tipo, sc)
        self._le_segmentos()
        if (self.turno_fim is not None and self.agora >= self.turno_fim
                and not self._terminal and self.ouvido.varrido < self.ouve_ate + FOLGA):
            pass                          # pela máquina de estados, o outro lado ainda fala
        elif (self.turno_fim is not None and self.agora >= self.turno_fim
                and self._ocupado() and self.agora < self.turno_fim + 60 * FS):
            pass                          # ainda chega som: o turno não acabou
        elif self.turno_fim is not None and self.agora >= self.turno_fim:
            if _DEPURA:
                self.anota(f"  fim de turno: terminal={self._terminal}, varrido "
                           f"{self.ouvido.varrido / FS:.2f}s, ouve_ate {self.ouve_ate / FS:.2f}s")
            turno, self.turno, self.turno_fim = self.turno, [], None
            self._turno(turno)
            self._terminal = False         # vale para o turno que acabou de ser tratado
        self._prazos()
        if self.agenda and self.agenda[0][0] <= self.agora and self.agora >= self.livre_em:
            _, x, eco = self.agenda.pop(0)
            a = self.agora
            self.livre_em = a + len(x) + self.lat_extra
            self._no_ar = (a, len(x), f"varredura de {eco}" if eco else " + ".join(self.tocando))
            if eco:
                # A varredura de saudação não é apagada da entrada: nenhuma
                # estação age sobre a do próprio tipo, e ouvir o próprio eco
                # é como se mede quanto o som demora de fato a sair.
                self.ouvido.cego(a, a)
                lead = getattr(x, 'lead', 0)
                self._eco = (eco, a + lead)
                if eco == 'resposta' and self.estado == 'respondendo':
                    k = x.caixa_marcas[0][1]
                    self._resp_tocadas.append((a + lead, k))
                    self._no_ar = (a, len(x), f"resposta no passo {k} ({CAIXA_POS[k]:.0%})")
                # A varredura não é apagada, e alta no próprio microfone ela
                # dá marca de quadro falsa.
                self._eco_janela = (a, a + self.lat_extra + len(x) + GUARDA)
                if eco == 'chamada':
                    self._chamou_fim = a + len(x)
            else:
                # Desde o instante em que toca, não desde a latência medida:
                # a de um alto-falante Bluetooth varia, e se sair mais cedo
                # que a medida a marca de abertura escapa da cegueira e a
                # estação abre "o turno do outro" sobre o próprio quadro. Com
                # as vezes, o outro lado não fala enquanto eu toco, então
                # começar cedo não custa nada.
                self.ouvido.cego(a, a + self.lat_extra + len(x) + GUARDA)
            if self.espera_ate is not None:
                self.espera_ate = max(self.espera_ate, self._fim_da_vez(a, len(x)))
            return x
        return None

    def atividade(self):
        """O que a estação está fazendo agora, em uma frase, para o painel:
        tocando o quê e quanto falta, esperando a vez, lendo um quadro que
        chega, ou esperando resposta e até quando. Antes da saudação, com o
        passo da caixa: o som ainda está sendo acertado."""
        a = self._atividade()
        if not self.saudado and self.estado in ('refina_ch', 'refina_ou'):
            k = self.caixa()
            quem = ("a do chamador" if self.estado == 'refina_ch' else "a do ouvinte")
            return f"refinando o som ({quem}): minha caixa em {CAIXA_POS[k]:.0%} -- {a}"
        if not self.saudado and self.estado == 'respondendo':
            k = self._resp_tocadas[-1][1] if self._resp_tocadas else 0
            return f"respondendo: passo {k} ({CAIXA_POS[k]:.0%}) -- {a}"
        if not self.saudado and self.estado in ('chamando', 'escutando', 'aguarda_ola'):
            k = self.caixa()
            quem = "janela" if self.papel == 'chamador' else "caixa no passo"
            return (f"acertando o som: {quem} {k} ({CAIXA_POS[k]:.0%})"
                    f"{', travada' if self.trava is not None else ''} -- {a}")
        return a

    def _atividade(self):
        t = self.agora
        if self._no_ar is not None:
            a, n, rot = self._no_ar
            if t < a + n + self.lat_extra:
                return f"tocando {rot} ({max(0, t - a) / FS:.1f} de {n / FS:.1f} s)"
        if self.agenda:
            q = self.agenda[0][0]
            return (f"na fila para tocar {' + '.join(self.tocando)}: "
                    f"espera a vez, {max(0, q - t) / FS:.0f} s")
        if self.pendente is not None:
            return (f"lendo um quadro que chega (marca ouvida há "
                    f"{max(0, t - self.pendente['pos']) / FS:.0f} s)")
        if self.turno:
            return f"o outro lado está falando: {len(self.turno)} segmento(s) neste turno"
        if self.espera_ate is not None:
            rep = f", {self.tentativas}ª repetição" if getattr(self, 'tentativas', 0) else ""
            return (f"esperando a resposta: se nada chegar em "
                    f"{max(0, self.espera_ate - t) / FS:.0f} s, repete{rep}")
        return "escutando"

    # --- recepção

    def _marca(self, pos, tipo, sc):
        if self._eco is not None and tipo == self._eco[0]:
            d = pos - self._eco[1]
            if -SPS <= d < 2 * FS:
                # O meu próprio eco. A latência que o runtime declarou já está
                # descontada em `_eco`; o que sobra é a que ninguém declarou
                # (um alto-falante Bluetooth declara menos do que tem). Sem
                # isto, a minha marca de fechamento sairia da janela de
                # cegueira e viraria a abertura de um quadro fantasma.
                novo = max(0, d)
                if abs(novo - self.lat_extra) > SPS:
                    self.anota(f"latência de saída medida no eco: +{novo / FS:.3f}s "
                               "além da declarada")
                self.lat_extra = novo
                self._eco = None
                return
        if tipo in (('chamada', 'nao_ch', 'sim_ch') if self.papel == 'chamador'
                    else ('resposta', 'nao_ou', 'sim_ou')):
            return                      # a do meu tipo, de um eco que já não espero
        if tipo == 'sonda':
            self._planeja_mic(pos)
        self.ouvido_em = pos
        # Antes de qualquer saudação ('escutando'), do outro lado só vêm
        # chamadas: não há quadro dele em que uma varredura possa casar por
        # engano. Lá uma chamada bem correlacionada vale pelo que é (a janela
        # de rms decide se ela é forte o bastante).
        # Fora do canal e antes do OLA nada do chamador além de varreduras
        # chega aqui, então a regra não vale em nenhum desses estados, com
        # qualquer correlação. Só em 'aguarda_ola' (o OLA do chamador já
        # pode estar no ar) ela exige uma chamada clara.
        so_chamadas = (not self.aberto
                       and (self.estado in ('escutando', 'respondendo', 'refina_ch', 'refina_ou')
                            or (self.estado == 'aguarda_ola' and sc >= CHAMADA_CLARA)))
        if tipo == 'chamada' and self.papel == 'ouvinte' and self._piso and not so_chamadas:
            r = self._rms_mediano(pos - FS // 2, pos)
            if r is not None and r > 2 * self._piso:
                # Uma chamada de verdade começa no silêncio: o chamador só chama
                # depois de esperar. Achada no meio de outro som, é a varredura
                # casando com um pedaço do quadro do chamador, e responder a ela
                # seria falar por cima dele.
                # Mediana de janelas de 20 ms, não a média do meio segundo: um
                # quadro ocupa o trecho todo; um estalo, não. E 2x o piso, não
                # 1,5x, porque o próprio silêncio passa um pouco do piso de 10%.
                self.anota(f"  chamada dentro de outro som ignorada ({sc:.2f}; "
                           f"antes dela rms {r:.4f}, piso {self._piso:.4f})")
                return
        if tipo == 'chamada':
            if self.papel == 'ouvinte' and self.estado in ('refina_ch', 'refina_ou'):
                self._chamada_no_refino(pos, sc)
                return
            if self.papel == 'ouvinte' and self.estado == 'respondendo':
                # Durante a minha escada rápida o chamador não chama de novo
                # (a janela dele cobre a escada inteira): uma chamada aqui é a
                # confirmação dele -- duas varreduras de chamada seguidas.
                self._confirmado(pos, sc)
                return
            if self.papel == 'ouvinte' and self.aberto and sc >= LIMIAR_RECOMECO and \
                    self.agora - self.progresso_em > SILENCIO_MORTO:
                # Recomeça, mas não responde a esta: se for mesmo o chamador
                # recomeçando, ele chama de novo em seguida.
                self.anota("chamada nova com o canal mudo há tempo; o outro lado "
                           "recomeçou -- recomeço também")
                self.reinicia()
                return
            if self.papel == 'ouvinte' and not self.aberto and self.saudado:
                # No meio da negociação. Com o limiar da chamada rente ao
                # falso, os tons da sondagem podem passar dele; se o outro
                # lado de fato recomeçou, os prazos de 'aguarda_*' devolvem
                # esta ponta à escuta e a chamada seguinte é ouvida.
                return
            if (self.papel == 'ouvinte' and not self.aberto and self._chamada_em is not None
                    and pos - self._chamada_em < 2 * len(MOLDES['chamada']) + RESP + ESPERA):
                # Logo depois de uma chamada (ou da confirmação, que são duas)
                # não vem outra: é eco, ou a segunda varredura da confirmação.
                if _DEPURA:
                    self.anota(f"  chamada cedo demais ignorada ({sc:.2f})")
                return
            if self.papel == 'ouvinte' and not self.aberto:
                r, pico = self._nivel(pos, tipo)
                if r is not None and r < CHAMADA_BOA_RMS and sc < OUVIDO_CORR:
                    # Ouvida, mas baixa: responder agora travaria o chamador
                    # num volume em que cada bit sai no limite. Calo; ele não
                    # ouve resposta e passa à janela seguinte, um passo acima.
                    self._chamada_em = pos
                    sobe = self._ajusta_mic(pico, 'chamada') > 0
                    self.anota(f"chamada ouvida baixa (rms {r:.3f} < {CHAMADA_BOA_RMS}, "
                               f"correlação {sc:.2f}, pico {pico:.2f}): não respondo; "
                               + ("subo o meu microfone e " if sobe else "")
                               + "o chamador sobe a caixa")
                    return
                self._chamada_em = pos
                self._ouvi_chamada = (pico, r)
                self._q_ant = self._qualidade(pos, tipo, sc)
                self._ajusta_mic(pico, 'chamada')
                rr = '?' if r is None else f'{r:.3f}'
                self.anota(f"chamada ouvida bem (correlação {sc:.2f}, rms {rr}, pico {pico:.2f})"
                           + (" -- chegou saturada" if pico >= PICO_SATURA else "")
                           + "; respondo em escada rápida, da caixa mais baixa para cima")
                # Perto ou longe não se decide aqui: decide o chamador, pela
                # resposta que ouvir, e o OLA traz a decisão.
                self._escada_resposta(pos)
        elif tipo == 'resposta':
            if self.estado == 'chamando' and self.papel == 'chamador':
                if self._chamou_fim is None or pos < self._chamou_fim:
                    # Resposta a uma chamada que ainda não terminou -- ou que
                    # nem saiu: é um falso, e seguir mandaria OLA a quem nunca
                    # o ouviu.
                    if _DEPURA:
                        self.anota(f"  resposta fora de hora ignorada ({sc:.2f} em {pos / FS:.2f}s)")
                    return
                r, pico = self._nivel(pos, tipo)
                self._ajusta_mic(pico, 'resposta')
                self._resp_na_janela += 1
                k = self.caixa()
                if (r is not None and r < CHAMADA_BOA_RMS and sc < OUVIDO_CORR
                        and self._resp_na_janela < 2 and not self._resposta_no_topo(pos)):
                    # Baixa: a escada dele ainda sobe dentro desta janela; a
                    # próxima resposta vem um passo acima. Não confirmo esta.
                    # Só a primeira: a 2ª detectada, ou uma que pelo tempo é o
                    # topo da escada dele, é o mais forte que ele tem: confirmo.
                    # Ele trava no passo que soou por último antes da
                    # confirmação, que é justamente essa (`_confirmado`).
                    self.anota(f"janela {k} ({CAIXA_POS[k]:.0%}): resposta baixa (rms {r:.3f}, "
                               f"correlação {sc:.2f}, pico {pico:.2f}); espero a próxima da escada dele")
                    return
                # Ouvida: confirmo já, e travo neste passo -- ele me ouviu
                # aqui e não no passo de baixo (quem ouve responde sempre).
                self._toca([self._confirmacao()], espera=False, guarda_turno=False, eco='chamada')
                self._trava(k, f"janela {k} ({CAIXA_POS[k]:.0%}): o ouvinte respondeu; confirmo")
                self.anota(f"minha caixa fica em {CAIXA_POS[k]:.0%}: é o que o outro lado precisa "
                           "para me ouvir; a dele é a dele")
                self._ouvi_resp = (pico, r)
                self._q_ant = self._qualidade(pos, tipo, sc)
                rr = '?' if r is None else f'{r:.3f}'
                self.anota(f"resposta ouvida (correlação {sc:.2f}, limiar {LIMIARES['resposta']}, "
                           f"rms {rr}, pico {pico:.2f}); agora refino a minha caixa pelo som")
                # A escada dele pode ter mais uma resposta já no ar quando a
                # confirmação chega: a primeira chamada do refino sai depois
                # de um passo dela.
                self.estado = 'refina_ch'
                self._ref = dict(k_best=k, k_try=None, espera=None,
                                 quando=self.agora + len(self._confirmacao())
                                 + self._passo_resposta(LAT_SOM_OUTRO))
            elif self.estado == 'refina_ou' and self.papel == 'chamador':
                self._resposta_no_refino(pos, tipo, sc)
        elif tipo in ('sim_ou', 'nao_ou'):
            # Uma resposta da escada rápida (`resposta`) já não conta aqui:
            # "melhorou" tem marca própria. As duas são guardadas e a mais
            # forte decide depois de `ASSENTA` (`_decide_votos`).
            if (self.estado == 'refina_ch' and self.papel == 'chamador'
                    and self._ref['espera'] is not None and pos >= (self._chamou_fim or 0)):
                self._ref.setdefault('votos', []).append((pos, tipo, sc))
        elif tipo in ('sim_ch', 'nao_ch'):
            if self.estado == 'refina_ou' and self.papel == 'ouvinte' and self._ref['espera'] is not None:
                self._ref.setdefault('votos', []).append((pos, tipo, sc))
        # quadro e sonda viram segmentos, lidos quando o áudio deles chegar

    def _resposta_no_topo(self, pos):
        """Pelo tempo desde o fim da minha chamada, a resposta em `pos` é a
        do último passo da escada rápida dele? A escada começa ~0,1 s depois
        de ele varrer a chamada; cada passo leva `_passo_resposta`."""
        if self._chamou_fim is None:
            return False
        d = self._passo_resposta(LAT_SOM_OUTRO)
        t0 = (self._chamou_fim + self.lat_extra + VARRE_ATRASO + int(0.1 * FS)
              + RESP_LEAD + LAT_SOM_OUTRO)
        return (pos - t0) / d >= N_POS - 1.5

    def _passo_resposta(self, lat_minha=None):
        """Intervalo entre duas respostas da escada rápida. Do instante em que
        a resposta j sai da minha caixa ao começo da confirmação chegando aqui
        vão no máximo: a varredura (0,2 s), o atraso da varredura de marcas do
        chamador, a latência de saída dele -- e metade da incerteza da minha,
        se ainda não a medi. O intervalo é maior que isso, com folga, para que
        a confirmação sempre chegue antes de a resposta j+1 soar: assim a
        última resposta que soou antes dela é a que foi ouvida, sem
        ambiguidade. ~2,4 s com os valores atuais."""
        incerta = LAT_SOM_OUTRO if (lat_minha is None and self.lat_extra == 0) else 0
        return (len(MOLDES['resposta']) + VARRE_ATRASO + LAT_SOM_OUTRO + incerta // 2
                + MARGEM_SOM)

    def _janela(self):
        """Uma janela do chamador, do início da chamada: a minha latência, a
        chamada, o atraso da varredura dele, o silêncio e a latência antes da
        primeira resposta, a escada rápida inteira dele, e a última resposta
        chegando e sendo varrida aqui. ~16 s com os valores atuais."""
        d = self._passo_resposta(LAT_SOM_OUTRO)
        return (self.lat_extra + len(MOLDES['chamada']) + VARRE_ATRASO + RESP_LEAD
                + LAT_SOM_OUTRO + (N_POS - 1) * d + len(MOLDES['resposta'])
                + VARRE_ATRASO + MARGEM_SOM)

    def _confirmacao(self):
        """Duas varreduras de chamada a 0,1 s: o ouvinte em escada rápida não
        espera chamada nova antes do fim da janela, e a segunda marca a
        separa de um eco."""
        g = GANHOS_CHAMADA[self.esforco][0]
        x = np.concatenate([MOLDES['chamada'] * g, np.zeros(CONFIRMA_GAP),
                            MOLDES['chamada'] * g]).view(Parte)
        x.rotulo = "confirmação (duas chamadas)"
        return x

    def _escada_resposta(self, pos):
        """Ouvi uma chamada bem em `pos`: respondo em todos os passos da
        caixa, do mais baixo para cima, um a cada `_passo_resposta`. Cada
        resposta leva o próprio passo (`caixa_marcas`), trocado no silêncio
        que a antecede; a escada para quando a confirmação chega."""
        d = self._passo_resposta()
        g = self._ganho_chamada()
        t0 = max(self.agora, pos + len(MOLDES['chamada'])) + int(0.1 * FS)
        self.agenda = []
        for j in range(N_POS):
            x = np.concatenate([np.zeros(RESP_LEAD), MOLDES['resposta'] * g]).view(Parte)
            x.caixa_marcas = [(0, j)]
            x.lead = RESP_LEAD
            x.rotulo = f"resposta no passo {j} ({CAIXA_POS[j]:.0%})"
            self.agenda.append((t0 + j * d, x, 'resposta'))
        self.tocando = [f"escada de respostas, {N_POS} passos a cada {d / FS:.1f}s"]
        self.anota(f"toco em {t0 / FS:.1f}s: " + self.tocando[0])
        self._resp_tocadas = []
        self._resp_fim = t0 + (N_POS - 1) * d + RESP_LEAD + self.lat_extra + d
        self.estado = 'respondendo'
        self.espera_ate = None

    def _confirmado(self, pos, sc):
        """A confirmação do chamador chegou em `pos`: a última resposta que
        soou antes dela (sair da caixa + a varredura) é a que ele ouviu."""
        L = len(MOLDES['resposta'])
        ouvidas = [k for a, k in self._resp_tocadas if a + self.lat_extra + L <= pos]
        if not ouvidas:
            self.anota(f"  chamada durante a escada antes de qualquer resposta soar ({sc:.2f}); ignoro")
            return
        k = ouvidas[-1]
        self.agenda = []                   # a escada para aqui
        self._chamada_em = pos             # a 2ª varredura da confirmação não é chamada nova
        self._resp_passo = k
        self._trava(k, f"confirmação ouvida depois do passo {k} ({CAIXA_POS[k]:.0%}): travo aí")
        # Agora o chamador refina a caixa dele: chama um passo acima, e eu
        # digo se melhorou em relação à chamada que abriu a minha escada.
        self.estado = 'refina_ch'
        self._dupla = None
        # A 1ª chamada de refino dele sai um passo da minha escada depois da
        # confirmação; sem ela, a vez é minha (ver `_prazos_refino`).
        self.espera_ate = (self.agora + self._passo_resposta() + len(MOLDES['chamada'])
                           + 2 * self._volta_ref())

    # --- refino pelo som

    def _qualidade(self, pos, tipo, sc):
        """(correlação, SNR em dB sobre o piso, rms) da varredura `tipo` em `pos`.
        Sem piso estimado (ele zera a cada troca de microfone), o piso é o do
        próprio trecho antes da varredura: mediana de janelas de 20 ms."""
        r, _ = self._nivel(pos, tipo)
        piso = self._piso
        if not piso:
            x = self.ouvido.trecho(max(0, pos - FS // 2), pos)
            n = int(0.02 * FS)
            if len(x) >= n:
                w = x[:len(x) // n * n].reshape(-1, n)
                piso = float(np.median(np.sqrt(np.mean(w ** 2, axis=1)))) or None
        snr = (20 * np.log10(r / piso) if r and piso else None)
        return (float(sc), snr, r)

    @staticmethod
    def _melhor(q, q0):
        """Abaixo do alvo de quadro (rms < CHAMADA_BOA_RMS no receptor) o
        degrau travado é só o piso do que se detecta, não o de que os quadros
        precisam. Aí só "claramente pior" para a subida; no alvo ou acima, vale
        o "claramente melhor" de sempre."""
        if q0 is None:
            return True
        dc = q[0] - q0[0]
        ds = (q[1] - q0[1]) if q[1] is not None and q0[1] is not None else None
        if q0[2] is not None and q0[2] < CHAMADA_BOA_RMS:
            return not (dc < -REF_CORR or (ds is not None and ds < -REF_SNR))
        if abs(dc) > REF_CORR:
            return dc > 0
        return ds is not None and ds >= REF_SNR

    @staticmethod
    def _por_que(q0):
        return (" (abaixo do alvo: subo enquanto não piorar)"
                if q0 is not None and q0[2] is not None and q0[2] < CHAMADA_BOA_RMS else "")

    @staticmethod
    def _desc_q(q):
        return f"correlação {q[0]:.2f}" + ("" if q[1] is None else f", {q[1]:.0f} dB sobre o piso")

    def _varredura(self, tipo, k, eco, rotulo):
        """Uma varredura de saudação no passo de caixa `k`, com o silêncio
        antes em que a caixa troca; vai para a agenda já."""
        g = GANHOS_CHAMADA[self.esforco][0]
        x = np.concatenate([np.zeros(RESP_LEAD), MOLDES[tipo] * g]).view(Parte)
        x.caixa_marcas = [(0, k)]
        x.lead = RESP_LEAD
        x.rotulo = rotulo
        self.agenda = [(self.agora, x, eco)]
        self.tocando = [rotulo]
        return len(x)

    def _volta_ref(self):
        """Quanto esperar a resposta a uma varredura de refino: ela sair
        (a minha latência), o outro varrer e decidir (inclusive esperar a 2ª
        varredura de uma dupla), a resposta sair de lá e eu varrê-la."""
        return (self.lat_extra + VARRE_ATRASO + DUPLA_ATE + RESP_LEAD + LAT_SOM_OUTRO
                + len(MOLDES['nao_ch']) + VARRE_ATRASO + MARGEM_SOM)

    def _decide_votos(self, r):
        """"Melhorou" e "não melhorou" ouvidos na mesma volta: fica o mais
        forte, e na dúvida (diferença menor que REF_CORR) "não melhorou" --
        voltar um passo custa pouco, subir à toa satura a caixa."""
        votos = r.pop('votos')
        sim = max((sc for _, t, sc in votos if t.startswith('sim')), default=0.0)
        nao = max((sc for _, t, sc in votos if t.startswith('nao')), default=0.0)
        melhor = sim > nao + REF_CORR if nao else sim > 0
        desc = f"melhorou {sim:.2f}, não melhorou {nao:.2f}"
        if sim and nao:
            self.anota(f"refino: ouvi as duas respostas ({desc}); fico com "
                       + ("melhorou" if melhor else "não melhorou"))
        if self.papel == 'chamador':
            self._ref_resultado(melhor, votos[0][0], 'nao_ou', nao)
        else:
            self._ref_ou_resultado(melhor, f"o chamador disse que {'melhorou' if melhor else 'não melhorou'} ({desc})")

    def _ref_resultado(self, melhor, pos, tipo, sc):
        """Chamador, refinando a própria caixa: o ouvinte respondeu à chamada
        no passo k_try com "melhorou" (resposta) ou "não melhorou"."""
        r = self._ref
        k = r['k_try']
        r['espera'] = None
        if melhor:
            r['k_best'] = k
            self._trava(k, f"refino: a chamada a {CAIXA_POS[k]:.0%} saiu melhor para o ouvinte")
            self.anota(f"refino da minha caixa: {CAIXA_POS[k]:.0%} melhorou para ele; subo mais")
            r['quando'] = self.agora + int(0.3 * FS)
            return
        kb = r['k_best']
        self.anota(f"refino da minha caixa: {CAIXA_POS[k]:.0%} "
                   + ("não melhorou para ele" if tipo else "ficou sem resposta")
                   + f"; volto a {CAIXA_POS[kb]:.0%} e travo")
        self._fim_ref_ch()

    def _fim_ref_ch(self):
        kb = self._ref['k_best']
        self.trava = kb
        self._pede_caixa(kb)
        # A vez dele: duas chamadas, como a confirmação.
        x = self._confirmacao()
        x.rotulo = "a vez do ouvinte refinar (duas chamadas)"
        x.caixa_marcas = [(0, kb)]
        self.agenda = [(self.agora + int(0.3 * FS), x, 'chamada')]
        self.tocando = [x.rotulo]
        self.anota(f"minha caixa fica em {CAIXA_POS[kb]:.0%}; agora o ouvinte refina a dele")
        self.estado = 'refina_ou'
        # Cobre também o ouvinte que não ouviu a dupla e só começa pelo
        # silêncio (`espera_ate` dele, ~2 voltas depois da última resposta).
        self._ref = dict(fim=self.agora + len(x) + 5 * self._volta_ref(), ola_em=None)

    def _resposta_no_refino(self, pos, tipo, sc):
        """Chamador, com o ouvinte refinando a caixa dele: cada resposta vem
        um passo acima; digo se melhorou em relação à anterior."""
        r = self._ref
        if r['ola_em'] is not None or pos < (self._chamou_fim or 0):
            return
        q = self._qualidade(pos, tipo, sc)
        _, pico = self._nivel(pos, tipo)
        self._ajusta_mic(pico, 'resposta')
        if self._melhor(q, self._q_ant):
            self.anota(f"refino da caixa do ouvinte: {self._desc_q(q)} contra "
                       f"{self._desc_q(self._q_ant)} -- melhorou{self._por_que(self._q_ant)}; digo a ele")
            self._q_ant = q
            self._varredura('sim_ch', self.caixa(), 'sim_ch', "melhorou")
            r['fim'] = self.agora + 2 * self._volta_ref()
            return
        self.anota(f"refino da caixa do ouvinte: {self._desc_q(q)} contra "
                   f"{self._desc_q(self._q_ant)} -- não melhorou; digo a ele, e depois o OLA")
        self._varredura('nao_ch', self.caixa(), 'nao_ch', "não melhorou")
        r['ola_em'] = self.agora + self._volta_ref()

    def _chamada_no_refino(self, pos, sc):
        """Ouvinte: uma chamada no refino é uma de três coisas -- a do
        chamador um passo acima (uma só), a vez de eu refinar (duas a 0,3 s),
        ou "melhorou" quando sou eu que subo. Uma só ou dupla só se sabe
        depois de DUPLA_ATE; decide-se em `_prazos`."""
        if self._chamada_em is not None and 0 < pos - self._chamada_em <= DUPLA_ATE:
            return                         # a 2ª varredura da confirmação
        if self.estado == 'refina_ou':
            return                         # "melhorou" agora é `sim_ch`
        if self._dupla is not None and 0 < pos - self._dupla[0] <= DUPLA_ATE:
            self._dupla = None
            self.agenda = []
            self.anota("duas chamadas: a caixa do chamador está travada; agora refino a minha")
            self.estado = 'refina_ou'
            self._ref = dict(k_best=self.trava, k_try=None, espera=None,
                             quando=self.agora + int(0.3 * FS))
            return
        if self._dupla is not None:
            # A pendente ficou sozinha: era simples, e esta é outra.
            p0, s0 = self._dupla
            self._dupla = None
            self._chamada_simples(p0, s0)
        self._dupla = (pos, sc)

    def _chamada_simples(self, pos, sc):
        """Ouvinte: a chamada do chamador um passo acima. Comparo com a
        anterior e respondo "melhorou" ou não, na minha caixa travada."""
        q = self._qualidade(pos, 'chamada', sc)
        _, pico = self._nivel(pos, 'chamada')
        self._ajusta_mic(pico, 'chamada')
        if self._melhor(q, self._q_ant):
            self.anota(f"refino da caixa do chamador: {self._desc_q(q)} contra "
                       f"{self._desc_q(self._q_ant)} -- melhorou{self._por_que(self._q_ant)}; respondo")
            self._q_ant = q
            self._varredura('sim_ou', self.trava, 'sim_ou', "melhorou")
        else:
            self.anota(f"refino da caixa do chamador: {self._desc_q(q)} contra "
                       f"{self._desc_q(self._q_ant)} -- não melhorou; digo a ele")
            self._varredura('nao_ou', self.trava, 'nao_ou', "não melhorou")
        # A próxima chamada de refino dele vem logo depois da minha resposta.
        # Sem ela, ele parou e me deu a vez -- mesmo que eu não tenha ouvido
        # a dupla que diz isso.
        self.espera_ate = self.agora + len(MOLDES['nao_ou']) + 2 * self._volta_ref()

    def _ref_ou_resultado(self, melhor, porque):
        """Ouvinte, refinando a própria caixa: a resposta no passo k_try."""
        r = self._ref
        k = r['k_try']
        r['espera'] = None
        if melhor:
            r['k_best'] = k
            self._trava(k, f"refino: a resposta a {CAIXA_POS[k]:.0%} saiu melhor para o chamador")
            r['quando'] = self.agora + int(0.3 * FS)
            return
        kb = r['k_best']
        self.anota(f"refino da minha caixa: {CAIXA_POS[k]:.0%} -- {porque}; volto a "
                   f"{CAIXA_POS[kb]:.0%}, travo e espero o OLA")
        self._fim_ref_ou()

    def _fim_ref_ou(self):
        kb = self._ref['k_best']
        self.trava = kb
        self._pede_caixa(kb)
        self.estado = 'aguarda_ola'
        self.espera_ate = self.agora + OLA_ESPERA_TOTAL

    def _prazos_refino(self):
        """Os tempos do refino, nos dois papéis. True se tratou o estado."""
        if self.estado not in ('refina_ch', 'refina_ou'):
            return False
        if self.agenda:
            return True
        r = getattr(self, '_ref', None)
        if r is not None and r.get('votos') and self.ouvido.varrido >= r['votos'][0][0] + ASSENTA:
            self._decide_votos(r)
            return True
        if self.papel == 'chamador' and self.estado == 'refina_ch':
            if r['espera'] is None and self.agora >= r['quando']:
                k = r['k_best'] + 1
                if k > N_POS - 1:
                    self.anota(f"refino da minha caixa: já no passo mais alto "
                               f"({CAIXA_POS[r['k_best']]:.0%})")
                    self._fim_ref_ch()
                    return True
                r['k_try'] = k
                n = self._varredura('chamada', k, 'chamada', f"chamada de refino a {CAIXA_POS[k]:.0%}")
                self.anota(f"refino da minha caixa: chamo a {CAIXA_POS[k]:.0%}; o ouvinte diz se melhorou")
                r['espera'] = self.agora + n + self._volta_ref()
            elif r['espera'] is not None and self.agora > r['espera'] and not r.get('votos'):
                self._ref_resultado(False, None, None, None)
        elif self.papel == 'chamador':
            if r['ola_em'] is not None and self.agora >= r['ola_em'] or self.agora > r['fim']:
                if r['ola_em'] is None:
                    self.anota("refino da caixa do ouvinte: nada mais dele; sigo para o OLA")
                self._comeca_ola()
        elif self.estado == 'refina_ch':
            if self._dupla is not None and self.ouvido.varrido >= self._dupla[0] + DUPLA_ATE:
                pos, sc = self._dupla
                self._dupla = None
                self._chamada_simples(pos, sc)
            elif self.espera_ate is not None and self.agora > self.espera_ate:
                self.anota("refino: o chamador parou de chamar; a vez é minha, refino a minha caixa")
                self.estado = 'refina_ou'
                self._ref = dict(k_best=self.trava, k_try=None, espera=None,
                                 quando=self.agora + int(0.3 * FS))
        else:
            if r['espera'] is None and self.agora >= r['quando']:
                k = r['k_best'] + 1
                if k > N_POS - 1:
                    self.anota(f"refino da minha caixa: já no passo mais alto "
                               f"({CAIXA_POS[r['k_best']]:.0%}); espero o OLA")
                    self._fim_ref_ou()
                    return True
                r['k_try'] = k
                n = self._varredura('resposta', k, 'resposta', f"resposta de refino a {CAIXA_POS[k]:.0%}")
                self.anota(f"refino da minha caixa: respondo a {CAIXA_POS[k]:.0%}; o chamador diz se melhorou")
                r['espera'] = self.agora + n + self._volta_ref()
            elif r['espera'] is not None and self.agora > r['espera'] and not r.get('votos'):
                self._ref_ou_resultado(False, "sem resposta do chamador")
        return True

    def _comeca_ola(self):
        """Saudação depois do som acertado e refinado: sempre no degrau lento
        (20 baud), o mais lento confiável -- a velocidade só se aprende
        subindo, na sondagem. Falhando, a escada sobe a caixa e desce a 10."""
        self.k_ola, self._ola_subiu = OLA_LONGE_DESDE, False
        self._ab_da_escada("o OLA sai no lento")
        self.sessao = int(self.rng.integers(1, 256))
        self.estado = 'ola'
        self.tentativas = 0
        self.turno_ultimo = self.agora
        self.anota(f"som acertado e refinado; OLA em {ABORDAGENS[self.ab].nome}")
        self._envia_ola()

    def _ajusta_mic(self, pico, oque):
        """Ganho do meu microfone, antes do contrato: o pico ouvido de `oque`
        fora de MIC_ALVO move o microfone um passo (`volume.MIC_POS`), e o
        runtime faz a troca já. Devolve +1, -1 ou 0. Os dois lados mexem cada
        um no seu: o outro não sabe que mexi, e não precisa."""
        if pico is None:
            return 0
        if pico < MIC_ALVO[0] and not self.mic_no_teto:
            d = 1
        elif pico > MIC_ALVO[1] and not self.mic_no_piso:
            d = -1
        else:
            return 0
        self.pedidos_mic_passo.append((self.agora, d, float(pico)))
        # O piso foi medido no volume de antes; com o microfone em outro
        # passo ele muda junto, e o velho faria o silêncio parecer "outro som".
        self._niveis, self._piso = [], None
        return d

    def _corrige_caixa(self, pico, porque):
        """O outro lado disse com que pico me ouviu: levo a MINHA caixa para
        ele receber ~0,3. A sondagem depois anda em volta da trava corrigida."""
        if pico is None or self.trava is None:
            return
        k = self.trava
        if pico >= CAIXA_PICO_ALTO:
            k2 = max(0, k - 1 - int(20 * np.log10(pico / CAIXA_PICO_ALTO) // 6))
        elif pico < CAIXA_PICO_BAIXO:
            # Um passo por 6 dB abaixo do alvo, no mínimo 2: um passo por OLA
            # repetido seria devagar demais para a sessão.
            k2 = min(N_POS - 1, k + max(2, int(np.ceil(20 * np.log10(CAIXA_PICO_BAIXO / max(pico, 1e-4)) / 6))))
        else:
            k2 = k
        self.anota(f"[volume] o outro lado me ouviu com pico {pico:.2f} ({porque}): "
                   f"caixa {CAIXA_POS[k]:.0%} → {CAIXA_POS[k2]:.0%}")
        if k2 != k:
            self.trava = k2
            self._pede_caixa(k2)

    def _planeja_mic(self, p1):
        """Uma sondagem do outro lado começou em `p1`: na metade dela o meu
        microfone desce um passo (`troca_mic`), e no fim volta ao que eu uso.
        Tem de ser agendado na hora em que a marca é achada, não quando a
        sondagem for lida -- a leitura só vem depois do fim dela. A marca de
        fechamento dos tons, e qualquer outra dentro da mesma sondagem, não
        reagendam. De longe não há volta do microfone."""
        if self.esforco or self.estado not in ESPERA_SONDA:
            return
        dur = sonda_amostras(False)
        if self._plano_mic is not None and 0 <= p1 - self._plano_mic < dur:
            return
        self._plano_mic = p1
        self.pedidos_mic += [(p1 + a, k) for a, k in troca_mic(False)]
        self.pedidos_mic.append((p1 + dur + int(0.3 * FS), self.mic))
        if _DEPURA:
            self.anota(f"  sondagem em {p1 / FS:.2f}s: microfone varre em "
                       f"{[round((p1 + a) / FS, 1) for a, _ in troca_mic(False)]}s")

    def _alta(self, pos, tipo, sc):
        """A varredura correlacionou pouco mas chegou alta: é distorção, não
        distância, e o esforço de longe (mais ganho, mais devagar) só piora.
        Daí rms >= RMS_ALTA e >= SNR_ALTA dB: as duas, porque um piso medido baixo
        demais não basta."""
        L = len(MOLDES[tipo])
        r = self._rms_valido(pos, pos + L)
        if r is None or not self._piso:
            return False
        snr = 20 * np.log10(r / self._piso)
        if r >= RMS_ALTA and snr >= SNR_ALTA:
            self.anota(f"{tipo} fraca ({sc:.2f} < {LIMIAR_LONGE}) mas alta (rms {r:.3f}, "
                       f"{snr:.0f} dB sobre o piso): distorção, não distância; fico de perto "
                       "-- a escada de volume é que acha a caixa certa")
            return True
        return False

    def _candidatos(self, marca=None):
        """Descritores que podem estar no ar agora, por ordem de chance; só
        os abertos pela `marca` dada, se houver -- a marca de abertura já diz
        se o quadro é de perto ou de longe."""
        c = list(ABORDAGENS)
        if self.contrato_rx is not None:
            # Só a escada do meu modo: as duas pontas rodam o mesmo, e um
            # comprimento de 5x2 que caísse perto de um de 16-FSK só custaria
            # leituras a mais, nunca uma leitura errada (o CRC decide).
            for d in range(self.ultimo_degrau + 1):
                for n in (DADOS_N, CTRL_N):
                    c.append(self.contrato_rx.descritor(d, n, self.modo_dados))
        return [d for d in c if marca is None or d.marca == marca]

    def _le_segmentos(self):
        while True:
            if self.pendente is None:
                prox = [m for m in self.ouvido.marcas
                        if (m[1] in QUADROS or (m[1] == 'sonda' and self.estado in ESPERA_SONDA))
                        and m[0] >= max(self._cursor, self.turno_ultimo if not self.turno else 0)
                        and not self._eco_janela[0] <= m[0] < self._eco_janela[1]
                        and m[0] not in self._usadas]
                quero = self._quero()
                # Só entre as duas de quadro: a de sondagem não tem par, e
                # apagá-la por uma de quadro falsa ao lado faria a sondagem
                # inteira ser lida como quadros ilegíveis.
                prox = [m for m in prox if quero is None or m[1] == quero or m[1] not in QUADROS or not any(
                    o[1] == quero and abs(o[0] - m[0]) < FS // 2 for o in prox)]
                if not prox:
                    return
                if self.turno and self._terminal:
                    # O turno já acabou: o segmento que o fecha foi lido. Uma
                    # marca logo depois é de outro turno -- ou falsa -- e não
                    # entra neste.
                    return
                pos, tipo, _ = prox[0]
                if (quero is not None and tipo in QUADROS and tipo != quero
                        and self.ouvido.varrido < pos + FS // 2):
                    return                 # a marca que o estado espera pode estar logo atrás
                if tipo in QUADROS:
                    dur = max(d.amostras for d in self._candidatos(tipo))
                else:
                    dur = sonda_amostras(bool(self.esforco))
                self.pendente = dict(pos=pos, tipo=tipo, tentou=set(),
                                     prazo=pos + int(dur * 1.006) + SPS)
                if not self.turno:
                    # Abertura de um turno novo: ele dura o que a máquina de
                    # estados diz, leia-se o meio dele ou não (ver também
                    # `_segmento`, que corrige isto se a abertura era falsa).
                    self._turno_ini = pos
                    self.ouve_ate = pos + int(self._vez_do_outro() * 1.006)
                    if _DEPURA:
                        self.anota(f"  turno do outro aberto em {pos / FS:.2f}s ({tipo}), estado "
                                   f"{self.estado}: ouço até {self.ouve_ate / FS:.2f}s")
                self.turno_fim = None              # outro segmento do mesmo turno
            p = self.pendente
            if p['tipo'] in QUADROS:
                # O prazo vale sobre o áudio já varrido, não sobre o presente: a
                # varredura anda ~0,9 s atrás, e um prazo contado do presente leria
                # pela abertura antes de a marca de fechamento -- que cai justo no
                # fim do candidato mais longo -- ter sido procurada.
                if not self._tenta_quadro(p, final=self.ouvido.varrido >= p['prazo']):
                    return
            else:
                if self.agora < p['prazo']:
                    return
                self._le_sonda(p['pos'])
            self._usadas.add(p['pos'])
            self.pendente = None

    def _busca_dirigida(self):
        """Prazo vencido sem nada lido: procura a resposta entre as marcas de
        quadro abaixo do limiar (`LIMIAR_DIRIGIDO`) desde o meu turno. A que
        der um quadro que passe no CRC vira marca de verdade, e a leitura
        normal a pega na próxima volta. True se promoveu uma."""
        quero = self._quero()
        ini = max(self._cursor, self.turno_ultimo)
        cands = sorted((m for m in self.ouvido.fracas
                        if (quero is None or m[1] == quero) and m[0] >= ini and m[0] not in self._usadas
                        and not self._eco_janela[0] <= m[0] < self._eco_janela[1]
                        and m not in self.ouvido.marcas),
                       key=lambda m: -m[2])[:4]
        for pos, tipo, sc in cands:
            for d in self._candidatos(tipo):
                audio = self.ouvido.trecho(pos, pos + d.amostras + FS)
                corpo, _ = le_quadro(audio, 0, d, None)
                if corpo is None:
                    continue
                self.anota(f"busca dirigida: marca {tipo} fraca ({sc:.2f} < {LIMIARES[tipo]}) "
                           f"em {pos / FS:.2f}s dá {NOMES.get(corpo[0], corpo[0])} em {d.nome}; "
                           f"lendo")
                self.ouvido.marcas.append((pos, tipo, sc))
                self.ouvido.marcas.sort()
                return True
        if cands:
            self.anota(f"busca dirigida: {len(cands)} marca(s) {quero} fraca(s) desde "
                       f"{ini / FS:.1f}s, nenhuma dá quadro")
        return False

    def _fechamento(self, p1, dur, marca='quadro'):
        """A marca de quadro que fecha um quadro aberto em p1 com duração
        nominal `dur`, se houver uma onde deve estar (±0,4%)."""
        alvo = p1 + dur - len(MOLDES[marca])
        tol = int(0.004 * dur)
        for pos, tipo, _ in self.ouvido.marcas:
            if tipo == marca and pos != p1 and abs(pos - alvo) <= tol:
                return pos
        return None

    def _tenta_quadro(self, p, final):
        """Tenta ler o quadro pendente. Cada candidato é lido assim que a sua
        marca de fechamento chega -- um quadro de controle de 4 s não espera
        o prazo do quadro mais lento possível (10 s). No prazo, os que não
        tiveram fechamento são lidos só pela abertura, com relógio nominal.
        Devolve True quando o segmento está resolvido, lido ou não."""
        p1 = p['pos']
        if self._le_candidatos(p, nominal=final):
            return True
        # Uma marca de fechamento caiu exatamente onde um dos comprimentos
        # previstos termina e a leitura falhou: o quadro tem esse comprimento
        # e está ilegível. Esperar o prazo do candidato mais lento para tentar
        # os outros só pela abertura empilharia as repetições do outro lado.
        lidos_com_fecho = [q2 for (_, q2) in p['tentou'] if q2 is not None]
        if not final and not lidos_com_fecho:
            if not self._calou(p1, p['tipo']):
                return False
            # Calou antes do prazo: o quadro acabou, e cabe no que já chegou.
            # Ler pela abertura, com relógio nominal, como no prazo: a marca de
            # fechamento pode ter chegado fora da tolerância.
            if self._le_candidatos(p, nominal=True):
                return True
        # Nada passou no CRC. Conta como segmento ilegível -- quem fala sabe
        # que alguém falou -- e o turno espera um pouco por mais. O fim é
        # agora, não a marca de abertura: uma resposta contada da abertura
        # sairia por cima da repetição do outro lado, que não me ouve
        # enquanto toca.
        if _DEPURA:
            self.anota(f"  ilegível em {p1 / FS:.2f}s: final={final}, "
                       f"tentou={[(d.nome, None if q2 is None else round((q2 - p1) / FS, 3)) for d, q2 in p['tentou']]}")
        self._segmento(('ilegivel', p1), self.agora)
        return True

    def _le_candidatos(self, p, nominal):
        """Lê o quadro pendente em cada candidato: pela marca de fechamento
        quando ela chegou, e só pela abertura (relógio nominal) se `nominal`.
        Cada (candidato, fechamento) é tentado uma vez. True se um passou no
        CRC; nesse caso o segmento já foi registrado.

        Primeiro todos os que têm fechamento, depois os nominais, do mais
        longo ao mais curto. A ordem importa porque um quadro pode conter
        outro mais curto que passa no CRC: no 5x2 multicanal, `pair_map` põe
        a cópia 0 inteira nos primeiros símbolos, então a cabeça de um rep4 é
        um rep1 válido. Lido como rep1, o quadro "acabaria" antes do fim de
        verdade e a resposta sairia por cima de quem ainda está tocando. A
        marca de fechamento é a prova do comprimento; sem ela, errar para o
        mais longo só atrasa a resposta, errar para o mais curto colide."""
        p1 = p['pos']
        cands = self._candidatos(p['tipo'])
        lq = len(MOLDES[p['tipo']])
        tentativas = []
        for d in cands:
            p2 = self._fechamento(p1, d.amostras, p['tipo'])
            if p2 is not None and self.agora >= p2 + lq + SPS:
                tentativas.append((d, p2))
        if nominal:
            for d in sorted(cands, key=lambda d: -d.amostras):
                if self.agora >= p1 + d.amostras + SPS:
                    tentativas.append((d, None))
        for d, q2 in tentativas:
            if (d, q2) in p['tentou']:
                continue
            p['tentou'].add((d, q2))
            fim = p1 + int(d.amostras * 1.006) + SPS
            audio = self.ouvido.trecho(p1, fim)
            corpo, _ = le_quadro(audio, 0, d, None if q2 is None else q2 - p1)
            if corpo is not None:
                if q2 is not None:
                    self._usadas.add(q2)
                fq = (q2 + lq) if q2 is not None else p1 + d.amostras
                xq = self.ouvido.trecho(p1, fq)
                if len(xq):
                    self._nivel_quadro = (float(np.max(np.abs(xq))), float(np.sqrt(np.mean(xq ** 2))))
                    if self.contrato_rx is None and self._plano_mic is None:
                        self._ajusta_mic(self._nivel_quadro[0], f'quadro {d.nome}')
                self._segmento(('quadro', corpo, d, None), fq, inicio=p1)
                return True
        return False

    def _calou(self, p1, marca='quadro'):
        """O outro lado parou de tocar? Sem a marca de fechamento não se sabe
        o comprimento do quadro, e esperar o do candidato mais lento atrasa o
        veredito em até sete segundos -- tempo em que quem falou já desistiu
        de esperar e repete. A energia volta ao piso quando o quadro acaba,
        e isso não depende de bit nenhum."""
        # Só sobre o áudio que a varredura já cobriu: ela anda até ~0,9 s
        # atrás do presente, e um silêncio visto antes dela pode ser o que
        # vem logo depois de uma marca de fechamento ainda não achada.
        ate = self.ouvido.varrido
        curto = min(d.amostras for d in self._candidatos(marca))
        if ate < p1 + int(0.99 * curto):
            return False
        # E só sobre o que não é cegueira: o que a própria estação tocou é
        # zerado na entrada, e zero lido como silêncio declararia "calou" com
        # o outro lado tocando. Janela quase toda cega: não se sabe, e "não
        # sei" não é silêncio.
        ini = p1 + len(MOLDES[marca]) + HUSH
        ref = self._rms_valido(ini, ini + FS // 2)
        rec = self._rms_valido(ate - int(0.4 * FS), ate)
        if ref is None or rec is None:
            return False
        return rec < 0.3 * ref

    def _rms_mediano(self, a, b, janela=int(0.02 * FS)):
        """Mediana do rms em janelas de 20 ms fora da cegueira em [a, b):
        o nível de um som contínuo, que um estalo curto não levanta. None se
        mais da metade do trecho for cega."""
        inv = self.ouvido._invalido(a, b)
        if inv.mean() > 0.5:
            return None
        x = self.ouvido.trecho(a, b)
        v = [float(np.sqrt(np.mean(x[i:i + janela][~inv[i:i + janela]] ** 2)))
             for i in range(0, len(x) - janela + 1, janela)
             if (~inv[i:i + janela]).sum() > janela // 2]
        return float(np.median(v)) if v else None

    def _rms_valido(self, a, b):
        """rms das amostras fora da cegueira em [a, b), ou None se mais da
        metade do trecho for cega."""
        inv = self.ouvido._invalido(a, b)
        if inv.mean() > 0.5:
            return None
        x = self.ouvido.trecho(a, b)[~inv]
        return float(np.sqrt(np.mean(x ** 2)))

    def _le_sonda(self, p1):
        dur_tons = 2 * LQ + 2 * HUSH + _sonda_tons_nominal()
        alvo = p1 + dur_tons - LQ
        p2 = None
        for pos, tipo, _ in self.ouvido.marcas:
            if tipo == 'sonda' and pos != p1 and abs(pos - alvo) <= int(0.004 * dur_tons):
                p2 = pos
                self._usadas.add(pos)
        longe = bool(self.esforco)
        if _DEPURA:
            sm = [(round(pos / FS, 2), round(sc, 2)) for pos, tipo, sc in self.ouvido.marcas
                  if tipo == 'sonda' and abs(pos - p1) < 60 * FS]
            self.anota(f"  sonda em {p1 / FS:.2f}s, fechamento "
                       f"{'não achado' if p2 is None else f'em {p2 / FS:.2f}s'}; marcas de sonda {sm}")
        audio = self.ouvido.trecho(p1, p1 + sonda_amostras(longe) + FS)
        margem = mede_tons(audio, 0, None if p2 is None else p2 - p1, self.sessao)
        r = 1.0 if p2 is None else (p2 - p1) / (dur_tons - LQ)
        # As sondas da escada, em posições conhecidas depois da marca final.
        res, niveis = [], []
        inicios, vaos, _ = sonda_layout(longe)
        q = int(round(dur_tons * r))
        for i, d in enumerate(sonda_g(longe)):
            if d is None:
                res.append((0.5, False))
                niveis.append(None)
                continue
            q += int(vaos[i] * r)
            perto = [pos - p1 for pos, tipo, _ in self.ouvido.marcas
                     if tipo == d.marca and abs(pos - p1 - q) <= int(0.03 * FS)]
            ber, inteiro, nivel, marg = 0.5, False, None, None
            if perto:
                s1 = perto[0]
                self._usadas.add(p1 + s1)
                fecho = [pos - p1 for pos, tipo, _ in self.ouvido.marcas
                         if tipo == d.marca
                         and abs(pos - p1 - s1 - (d.amostras - d.lm)) <= int(0.004 * d.amostras)]
                p2q = fecho[0] if fecho else None
                if p2q is not None:
                    self._usadas.add(p1 + p2q)
                corpo, llr = le_quadro(audio, s1, d, p2q)
                if llr is not None:
                    conhecido = empacota(SONDA_G, self.sessao, carga_sonda_g(self.sessao, i), SONDA_N)
                    ref = fec.encode(conhecido, repeat=1)
                    k = min(len(ref), len(llr))
                    ber = float(np.mean((llr[:k] > 0) != (ref[:k] > 0))) if k else 0.5
                    inteiro = corpo == conhecido
                    if k:
                        marg = margem_llr(llr[:k], ref[:k])
                # O nível do corpo (entre as marcas): pico para o corte do
                # conversor, rms para ver se a caixa sobe o que pedimos.
                corpo_a = np.asarray(audio[s1 + d.lm + HUSH:s1 + d.amostras - d.lm - HUSH])
                if len(corpo_a):
                    nivel = (float(np.max(np.abs(corpo_a))), float(np.sqrt(np.mean(corpo_a ** 2))),
                             marg)
                q = s1
            q += int(d.amostras * r)
            res.append((ber, inteiro))
            niveis.append(nivel)
        fim = p1 + sonda_amostras(longe)
        if np.median(margem) < MARGEM_FORA and not any(ok for _, ok in res):
            # Os tons enviados chegaram mais fracos que os não enviados, e
            # nenhuma sonda de ganho foi achada: não é um canal ruim (num
            # canal ruim as margens caem para perto de zero, não abaixo), é
            # a sondagem lida a partir da marca errada. Conta como ilegível e
            # espera a repetição.
            self.anota(f"sondagem fora do lugar (margem mediana {np.median(margem):.0f} dB, "
                       f"nenhuma sonda de ganho achada); tratada como ilegível")
            self._segmento(('ilegivel', p1), fim)
            return fim
        # O microfone varreu a partir da marca que achei na hora; se a
        # sondagem lida abriu em outro lugar, as metades não são as que eu
        # penso, e o microfone fica fora da escolha.
        mic_ok = (not longe and self._plano_mic is not None
                  and abs(self._plano_mic - p1) <= int(0.05 * FS))
        self._segmento(('sonda', margem, res, longe, niveis, mic_ok), fim, inicio=p1)
        return fim

    def _segmento(self, ev, fim, inicio=None):
        if ev[0] != 'ilegivel' and inicio is not None:
            self._lido_ini = inicio        # abertura do último segmento lido
        if (inicio is not None and ev[0] != 'ilegivel'
                and all(e[0] == 'ilegivel' for e in self.turno)):
            # O primeiro segmento lido do turno é a abertura de verdade; as
            # ilegíveis antes dele podem ter sido marcas falsas. O prazo do
            # turno conta daqui, e só cresce. Só no lido: re-ancorar a cada
            # ilegível alongaria a espera a cada marca falsa.
            self.ouve_ate = max(self.ouve_ate, inicio + int(self._vez_do_outro() * 1.006))
        self.turno.append(ev)
        self.turno_ultimo = max(self.turno_ultimo, fim)
        # Lido, o quadro ocupa até o fim; ilegível, não se sabe o
        # comprimento, e as marcas que chegaram enquanto se tentava lê-lo
        # continuam valendo.
        self._cursor = ev[1] + 1 if ev[0] == 'ilegivel' else max(self._cursor, fim)
        if ev[0] == 'ilegivel' and self.turno and self._terminal:
            return                         # o turno já fechou no lido; ver `_le_segmentos`
        ultimo = (ev[0] == 'sonda'
                  or (ev[0] == 'quadro' and not abre(ev[1])[1]))
        self._terminal = ultimo
        self.turno_fim = self.agora if ultimo else self.agora + FOLGA

    def _vez_do_outro(self):
        """Quanto dura o turno que o outro lado manda em seguida, pelo estado
        em que estou. As duas pontas rodam a mesma máquina de estados, então
        isso se sabe sem ouvir nada -- e é o que separa as vezes de falar: uma
        marca perdida no meio do turno não o encerra cedo. Na dúvida, o mais
        longo possível: esperar demais só atrasa, falar cedo colide."""
        e = self.estado
        if e == 'ola':
            # O OLA_OK vem no degrau em que o outro lado leu o meu OLA: o mesmo.
            return ABORDAGENS[self.ab].amostras
        if self.papel == 'ouvinte' and not self.saudado:
            # Um OLA pode vir em qualquer degrau da escada: o turno dura o
            # mais longo deles.
            return max(ABORDAGENS[i].amostras for i in OLA_ESCADA)
        if e == 'sondando':
            return self.ctrl.amostras + GAP + sonda_amostras(bool(self.esforco))
        if e == 'aguarda_sonda':
            return sonda_amostras(bool(self.esforco))
        if e == 'aguarda_contrato':
            d = self.contrato_rx.descritor(self._degrau_inicial(self.contrato_rx),
                                           DADOS_N, self.modo_dados)
            return self.ctrl.amostras + GAP + d.amostras
        if self.aberto and self.contrato_rx is not None:
            # Um quadro só; o outro lado pode ter descido um degrau desde o
            # último que li (por silêncio, ou acompanhando o meu pedido).
            k = min(self.degrau_rx + 1, self.ultimo_degrau)
            return self.contrato_rx.descritor(k, DADOS_N, self.modo_dados).amostras
        return 0

    # --- prazos

    def _prazos(self):
        if self.encerrada:
            return
        if self.aberto and self.agora - self.progresso_em > 3 * SILENCIO_MORTO:
            # Nem a chamada de quem recomeçou foi ouvida com força (ver
            # LIMIAR_RECOMECO): sem nada legível por tanto tempo, recomeço
            # sozinho em vez de esperar para sempre num canal morto.
            self.anota("canal aberto sem nada legível há muito tempo; recomeçando")
            self.reinicia()
            return
        if (self.aberto and self.confirmado and self.papel == 'chamador'
                and self.pendente_tx is None and self.ocioso()
                and self.agora - self.ouvido_em > 2 * FS
                and (self.fila or self.outro_tem_mais
                     or (self.sondar_em is not None and self.agora >= self.sondar_em))):
            # Só o chamador abre uma troca no canal aberto; o ouvinte só
            # responde, e manda o que tem dentro da resposta. Com os dois
            # podendo começar, os dois começariam juntos. Ocioso, o chamador
            # ainda consulta de tempos em tempos, para o ouvinte poder falar.
            self.sondar_em = None
            self._toca(self._turno_dados(), apos=self.agora)
            return
        if self._prazos_refino():
            return
        if self.estado == 'chamando':
            if self.espera_ate is None and self.agora >= self.proxima_chamada and not self.agenda:
                g = self._ganho_chamada()
                k = self.caixa()
                self.anota(f"chamando com a caixa no passo {k} ({CAIXA_POS[k]:.0%})"
                           + (", travada" if self.trava is not None else ""))
                self.turno_ultimo = self.agora
                self._toca([MOLDES['chamada'] * g], espera=True, guarda_turno=False,
                           eco='chamada')
                # A janela inteira, não a espera de uma resposta só: dentro
                # dela o ouvinte sobe a escada rápida dele.
                self.espera_ate = self.agora + self._janela()
                self._resp_na_janela = 0
            elif (self.espera_ate is not None and self.agora > self.espera_ate
                  and not self.agenda):
                self.espera_ate = None
                # Depois da janela, mais um passo da escada dele antes de
                # chamar de novo: uma confirmação atrasada não pode cair em
                # cima da chamada seguinte.
                self.proxima_chamada = (self.agora + self._passo_resposta(LAT_SOM_OUTRO)
                                        + int(self.rng.uniform(0.0, 0.5) * FS))
                if self.trava is None:
                    k0 = self.caixa()
                    self.k_som = min(self.k_som + 1, len(CHAMADA_PASSOS) - 1)
                    k = self.caixa()
                    self.anota(f"janela {k0} ({CAIXA_POS[k0]:.0%}): sem resposta"
                               + (f"; subo para a janela {k} ({CAIXA_POS[k]:.0%})" if k != k0
                                  else "; já no passo mais alto"))
                    self._pede_caixa(k)
            return
        if self.estado == 'respondendo':
            if self.agora > self._resp_fim and not self.agenda:
                k = self._resp_tocadas[-1][1] if self._resp_tocadas else None
                self.anota("escada de respostas sem confirmação"
                           + (f" (até o passo {k}, {CAIXA_POS[k]:.0%})" if k is not None else "")
                           + "; espero a próxima chamada, um passo acima")
                self.estado = 'escutando'
            return
        if (self.espera_ate is None or self.agora <= self.espera_ate or self.agenda
                or self.pendente is not None or self.turno or self.agora < self.livre_em):
            return
        if self._ocupado():
            # Algo ainda soa: não repito por cima. O piso se adapta em ~2 min,
            # então um ruído novo e constante não trava isto para sempre.
            return
        if self._busca_dirigida():
            return                 # a resposta estava lá, sob o limiar; vai ser lida
        # Prazo vencido sem resposta.
        self.espera_ate = None
        if self.estado == 'ola':
            # Sem OLA_OK: primeiro mais som no mesmo degrau, depois mais
            # devagar com o som já subido.
            self.tentativas += 1
            d = ABORDAGENS[self.ab]
            k = self.caixa()
            if not self._ola_subiu and k < N_POS - 1:
                self._ola_subiu = True
                self.trava = k + 1
                self._pede_caixa(k + 1)
                self.anota(f"OLA sem resposta a {d.baud} baud; subo a caixa para "
                           f"{CAIXA_POS[k + 1]:.0%} e repito")
                self._envia_ola()
                return
            if self.k_ola + 1 < len(OLA_ESCADA):
                self.k_ola += 1
                self._ola_subiu = False
                self._ab_da_escada(f"OLA a {d.baud} baud sem resposta")
                self.anota(f"OLA sem resposta a {d.baud} baud; desço para "
                           f"{ABORDAGENS[self.ab].baud} baud ({ABORDAGENS[self.ab].nome}), "
                           f"caixa em {CAIXA_POS[self.caixa()]:.0%}")
                self._envia_ola()
                return
            self.anota(f"OLA sem resposta em todos os degraus; voltando a chamar com a caixa "
                       f"em {CAIXA_POS[k]:.0%} e o microfone onde está")
            self.reinicia(mantem_som=True)
        elif self.fim_tentativas and self.aberto and self.papel == 'chamador':
            if self.fim_tentativas >= FIM_TENTATIVAS:
                # Tudo já tinha sido entregue e confirmado antes do FIM; o
                # que se perdeu foi só a despedida. Não há o que esperar.
                self.anota(f"{self.fim_tentativas} FIM sem resposta; tudo já estava "
                           "entregue -- encerro")
                self.encerrada = True
                return
            self.fim_tentativas += 1
            self.anota(f"FIM sem resposta; repito ({self.fim_tentativas}/{FIM_TENTATIVAS})")
            self._toca(self._turno_fim(), apos=self.agora)
        elif self.estado == 'sondando' or (self.aberto and self.papel == 'chamador'):
            self._sem_resposta()
        elif self.estado in ('aguarda_sonda', 'aguarda_contrato') and self.tentativas < 3:
            # O turno do outro lado chegou ilegível (ou não chegou): ele vai
            # repeti-lo, que é o que faz quem fala ao não ouvir resposta.
            # Desistir aqui jogaria fora a saudação inteira.
            self.tentativas += 1
            self.anota(f"nada legível em '{self.estado}'; esperando a repetição "
                       f"({self.tentativas}/3)")
            self.espera_ate = self.agora + 2 * max(d.amostras for d in self._candidatos()) + ESPERA
        elif self.estado in ('aguarda_ola', 'aguarda_sonda', 'aguarda_contrato'):
            self.anota(f"nada mais em '{self.estado}'; voltando a escutar")
            self.reinicia(mantem_som=self.estado == 'aguarda_ola')

    def _sem_resposta(self):
        """O turno ficou sem resposta lida: conta, desce o degrau se for
        o caso e repete. Um caminho só, venha do prazo vencido ou de um
        turno ilegível que chegou depois dele, porque o ouvinte calado não
        fala sobre o que não leu e a escada precisa descer mesmo assim."""
        self.tentativas += 1
        self.sem_progresso += 1
        if self.tentativas > 8 or self.sem_progresso > 10:
            self.anota(f"sem resposta em '{self.estado}'; voltando ao começo")
            self.reinicia()
            return
        if self.aberto and self.confirmado and self.tentativas % 2 == 0 \
                and self.degrau_tx < self.ultimo_degrau:
            # Silêncio não é falha que o receptor possa contar: se nem a
            # marca foi achada, ele não sabe que falei. Quem desce o
            # degrau, então, sou eu; o receptor tenta todos os degraus
            # pelo comprimento do quadro, sem precisar ser avisado.
            self.degrau_tx += 1
            self.anota(f"duas repetições sem resposta; desço para {self.degraus[self.degrau_tx]}")
            if self.degrau_rx < self.ultimo_degrau:
                # O silêncio pode ser eu que não ouço o outro lado: o
                # ouvinte não repete por conta própria, então o pedido
                # para ele descer também tem de partir daqui.
                self.degrau_rx += 1
                self.anota(f"e peço {self.degraus[self.degrau_rx]} ao outro lado")
        self.anota(f"sem resposta em '{self.estado}'; repetindo o turno")
        # Já depois da vez inteira do ouvinte (o prazo é o fim dela), e
        # sem sorteio: só o chamador abre turnos, não há com quem
        # desempatar.
        self._toca(self._retransmite(), apos=self.agora)

    def _retransmite(self):
        if self.aberto and self.confirmado:
            # Reconstruído, não repetido: o ack e o pedido de degrau podem ter
            # mudado desde a última vez.
            return self._turno_dados(forcar=True)
        if self.aberto and self.contrato_tx is not None:
            # O turno do contrato ficou sem resposta. O CONTRATO foi na
            # abordagem de controle, e o outro lado pode tê-lo perdido mesmo
            # lendo os DADOS que vinham atrás. Repetido, ele vai no formato
            # do contrato que o outro lado mediu para este sentido e já
            # procura (`_candidatos`, CTRL_N).
            d = self.contrato_tx.descritor(self._degrau(), CTRL_N, self.modo_dados)
            return ([self._quadro(d, CONTRATO, self.contrato_rx.empacota() + self._byte_modo(),
                                  mais=True)]
                    + self._turno_dados(forcar=True, so_partes=True))
        return self.ultimo_turno

    # --- fases

    def _envia_ola(self):
        d = ABORDAGENS[self.ab]
        pr, rr = self._ouvi_resp or (None, None)
        self._toca([self._quadro(d, OLA, bytes([self.ab, self.esforco, VERSAO,
                                                  q_db(pr), q_db(rr)]))], depois=RESP)

    def _versao_ok(self, v):
        """Recusa a sessão com quem roda outro formato: não responde, e diz
        no diário por quê, a cada vez -- um silêncio sem explicação seria
        lido como canal ruim."""
        if v == VERSAO:
            return True
        self.versao_outro = v
        self.anota(f"AVISO: versão do protocolo diferente: {v} lá, {VERSAO} aqui; "
                   "recuso a sessão -- atualizar as duas pontas para o mesmo código")
        return False

    def _turno(self, eventos):
        if self.encerrada:
            # Despediu-se: não responde a mais nada.
            return
        quadros = [(abre(e[1]), e[2]) for e in eventos if e[0] == 'quadro'
                   and abre(e[1])[0] in NOMES]
        sondas = [e for e in eventos if e[0] == 'sonda']
        ruins = sum(1 for e in eventos if e[0] == 'ilegivel')
        tipos = [NOMES[q[0][0]] for q in quadros] + ['SONDA'] * len(sondas)
        self.anota(f"turno: {tipos or '-'}" + (f", {ruins} ilegível(is)" if ruins else ""))
        # Uma sessão nova do outro lado (OLA) é ouvida em qualquer estado,
        # menos com o canal aberto -- aí só se o OLA vier de outra sessão.
        for (tipo, mais, sess, carga), d in quadros:
            if tipo == OLA and self.papel == 'ouvinte' and (not self.aberto or sess != self.sessao):
                if not self._versao_ok(carga[2]):
                    return
                if self.aberto:
                    self.reinicia()
                self.sessao = sess
                # A abordagem vem pelo índice que o OLA carrega, e com ela o
                # esforço de quem chamou. Se qualquer uma das pontas se acha
                # longe, as duas passam a falar de longe daqui em diante: a
                # distância vale para os dois sentidos, e uma ponta que
                # respondesse de perto a quem fala de longe não seria ouvida.
                i = carga[0] if carga[0] < len(ABORDAGENS) else ABORDAGENS.index(d)
                # Perto/longe é do chamador, e vem no OLA; respondo no degrau
                # em que li.
                self._poe_esforco(int(bool(carga[1])), f"o OLA veio a {ABORDAGENS[i].baud} baud "
                                  f"e o chamador decidiu")
                repetido = self._ola_lido == (sess, i)
                if repetido and self.trava is not None:
                    # O mesmo OLA de novo: o meu OLA_OK não foi ouvido. Ele
                    # sobe a caixa dele; eu subo a minha, direto ao topo.
                    k = N_POS - 1
                    if k != self.trava:
                        self.trava = k
                        self._pede_caixa(k)
                        self.anota(f"o mesmo OLA de novo: o meu OLA_OK não foi ouvido; "
                                   f"subo a caixa direto ao topo, {CAIXA_POS[k]:.0%}")
                self._ola_lido = (sess, i)
                self.ctrl = ABORDAGENS[i]
                self.saudado = True
                if self._resp_passo is not None and self.trava is None:
                    # O chamador só manda OLA a uma resposta que ouviu bem:
                    # a última que toquei é a que serviu.
                    self._trava(self._resp_passo, "o OLA veio depois da minha resposta neste volume")
                # E o OLA diz com que pico ele ouviu a minha resposta -- uma
                # vez por sessão: a repetição traz o mesmo relato.
                if not repetido:
                    self._corrige_caixa(dq_db(carga[3]), "a minha resposta, pelo OLA")
                self.estado = 'aguarda_sonda'
                self.anota(f"OLA lido em {d.nome}; saudação concluída, respondendo OLA_OK "
                           f"em {self.ctrl.nome}")
                pc, rc = self._ouvi_chamada or (None, None)
                po, ro = self._nivel_quadro or (None, None)
                self._minha_vez([self._quadro(self.ctrl, OLA_OK, bytes(
                    [i, self.esforco, VERSAO, q_db(pc), q_db(rc), q_db(po), q_db(ro)]))])
                return
        quadros = [(q, d) for q, d in quadros if q[2] == self.sessao]
        tipos = {q[0]: (q, d) for q, d in quadros}

        if self.estado == 'ola':
            if OLA_OK in tipos:
                i, esf, v = tipos[OLA_OK][0][3][:3]
                if not self._versao_ok(v):
                    return
                if esf != self.esforco:
                    self._poe_esforco(int(bool(esf)), "o OLA_OK trouxe outra distância")
                # A abordagem em que o outro lado respondeu, que é a de longe
                # se qualquer um dos dois estiver longe.
                self.ctrl = ABORDAGENS[i] if i < len(ABORDAGENS) else tipos[OLA_OK][1]
                rel = tipos[OLA_OK][0][3]
                # Como o ouvinte me ouviu: o OLA, que saiu na caixa de agora;
                # a chamada só se o OLA não tiver nível.
                po, pc = dq_db(rel[5]), dq_db(rel[3])
                self._corrige_caixa(po if po is not None else pc,
                                    "o meu OLA, pelo OLA_OK" if po is not None
                                    else "a minha chamada, pelo OLA_OK")
                self.saudado = True
                self.tentativas = 0
                self.anota(f"OLA_OK lido em {self.ctrl.nome}; saudação concluída, sondando")
                self.estado = 'sondando'
                self._toca([self._sonda()], depois=RESP)
            return

        if self.estado == 'aguarda_sonda' or (self.estado == 'aguarda_contrato'
                                               and sondas and CONTRATO not in tipos):
            if sondas:
                self.contrato_rx = self._contrato_de(sondas[0], "chega")
                self.estado = 'aguarda_contrato'
                self.tentativas = 0
                self._minha_vez([self._quadro(self.ctrl, CONTRATO,
                                              self.contrato_rx.empacota() + self._byte_modo(),
                                              mais=True),
                                 self._sonda()])
            return

        if self.estado == 'sondando':
            if CONTRATO in tipos and sondas:
                self.contrato_tx = Contrato.abre(tipos[CONTRATO][0][3])
                self._confere_modo(tipos[CONTRATO][0][3])
                self.anota(f"contrato para o que EU envio: {self.contrato_tx.descreve()}")
                self.contrato_rx = self._contrato_de(sondas[0], "chega")
                self.estado = 'contratando'
                self.tentativas = 0
                self._abre_canal()
                self._toca([self._quadro(self.ctrl, CONTRATO, self.contrato_rx.empacota() + self._byte_modo(), mais=True)]
                           + self._turno_dados(forcar=True, so_partes=True), depois=RESP)
            return

        if self.estado == 'aguarda_contrato':
            if CONTRATO in tipos:
                self.contrato_tx = Contrato.abre(tipos[CONTRATO][0][3])
                self._confere_modo(tipos[CONTRATO][0][3])
                self.anota(f"contrato para o que EU envio: {self.contrato_tx.descreve()}")
                self._abre_canal()
                self._dados(quadros, ruins + (0 if DADOS in tipos or ACK in tipos else 1))
            return

        if self.aberto:
            if CONTRATO in tipos and self.papel == 'ouvinte':
                # O chamador repetiu o turno do contrato: a primeira resposta
                # se perdeu. Responder de novo.
                pass
            self._dados(quadros, ruins)

    def _contrato_de(self, sonda, sentido):
        _, margem, res, longe, niveis, mic_ok = sonda
        caixas, mics = sonda_caixas(longe), sonda_mics(longe)
        descs = sonda_g(longe)
        ne = n_escada(longe)
        vres = [(res[i][0], res[i][1], None if niveis[i] is None else niveis[i][2])
                for i in range(ne, len(res))]
        res, niveis, caixas, mics, descs = res[:ne], niveis[:ne], caixas[:ne], mics[:ne], descs[:ne]
        if not mic_ok:
            # Sem saber em que metade o microfone estava, as duas metades
            # contam como o microfone de partida.
            mics = (0,) * len(mics)
        marg = [None if n is None else n[2] for n in niveis]
        picos = [None if n is None else n[0] for n in niveis]
        # A caixa, por subida na margem, para cada microfone; depois o
        # microfone, pela margem da caixa que cada um escolheu. O erro de bit
        # sozinho satura em 0% e não diz se o passo de cima seria mais limpo.
        escolha = {}
        for m in sorted(set(mics)):
            idx = sorted((j for j in range(ne) if mics[j] == m and descs[j] is not None),
                         key=lambda j: caixas[j])
            i, motivo = sobe_caixa([(caixas[j], res[j][0], res[j][1], marg[j], picos[j])
                                    for j in idx])
            escolha[m] = (None if i is None else idx[i], motivo)
        j, mic = escolha.get(0, (None, ''))[0], 0
        for m, (jm, _) in escolha.items():
            if m and jm is not None and (j is None or marg[jm] >= marg[j] + MARGEM_MIC_MIN):
                j, mic = jm, m
        if j is None:
            # Nada decodificou: a escolha pelo erro de bit (de
            # longe, o passo mais alto).
            c0, mic, j = decide(margem, res, caixas, mics, picos)
            k, motivo_caixa = c0.ganho, "nenhum passo decodificou; pelo erro de bit"
        else:
            k, motivo_caixa = caixas[j], escolha[mic][1]
        d, vtxt = escolhe_velocidade(vres) if vres else (None, '')
        if d is None:
            # Nenhum bloco da bateria passou: o mais robusto que há.
            d = MUITO_LENTO
        c = Contrato(tuple(escolhe_16(margem)), escolhe_par(margem), k, d)

        def celula(i):
            b, ok = res[i]
            n = niveis[i]
            nv = "" if n is None else f", pico {n[0]:.2f} rms {n[1]:.3f}"
            mg = "não lida" if marg[i] is None else f"margem {marg[i]:.1f} dB"
            return (f"caixa {caixas[i]}{' mic-8' if mics[i] else ''}: {mg}{nv}, "
                    f"erro {b * 100:.0f}%{'*' if ok else ''}")
        bers = "; ".join(celula(i) for i in range(len(res)) if descs[i] is not None)
        self.anota(f"sondagem lida: {bers}; margens dos tons "
                   f"{np.round(margem).astype(int).tolist()} dB")
        self.anota(f"caixa do outro lado: passo {k} -- {motivo_caixa}")
        if vtxt:
            self.anota(vtxt)
        self.anota(f"contrato para o que me {sentido}: {c.descreve()}")
        if not longe and mic_ok and mic != self.mic:
            self.mic = mic
            self.pedidos_mic.append((self.agora, mic))
            self.anota(f"microfone: passo {mic} ({'8 dB abaixo do de partida' if mic else 'o de partida'}), "
                       "o que leu melhor a escada")
        elif not longe and not mic_ok:
            self.anota("microfone fora da escolha: a sondagem lida não abriu na marca em que ele varreu")
        if c.ganho == 0:
            # A borda da escada, não um ótimo cercado: o outro lado pode
            # precisar de menos ainda, e isso a tabela dele não alcança.
            self.anota("AVISO: venceu o passo mais baixo da caixa do outro lado; "
                       "a escada dele (AUDIOFSK_CAIXA) pode precisar descer")
        return c

    # --- modo de dados

    def _byte_modo(self):
        """O código do meu modo de dados, no byte que sobra do CONTRATO.
        Em 'auto' é 0, o mesmo preenchimento dos outros quadros."""
        return bytes([MODOS_DADOS.index(self.modo_dados)])

    def _confere_modo(self, carga):
        """Escreve no diário se o outro lado roda outro modo de dados. Não
        negocia: o modo é escolhido por quem mede, nas duas pontas, e uma
        divergência tem de aparecer como divergência -- os dados de um lado
        não vão ser lidos pelo outro, e sem esta linha isso pareceria um
        canal ruim."""
        k = carga[5] if len(carga) > 5 else 0
        outro = MODOS_DADOS[k] if k < len(MODOS_DADOS) else f'desconhecido ({k})'
        if outro != self.modo_dados:
            self.anota(f"AVISO: o outro lado usa modo de dados {outro}, eu {self.modo_dados}; "
                       "os quadros de dados não vão se entender -- rodar as duas pontas "
                       "com o mesmo --modo-dados")
        else:
            self.anota(f"modo de dados confirmado nas duas pontas: {outro}")

    def _degrau_inicial(self, contrato):
        """Onde a escada do meu modo começa. Em 'auto' é o degrau do
        contrato. No 16-FSK fixo, o do contrato limitado ao rep4 (a sondagem
        mede justamente o 16-FSK). Nos outros, o primeiro: a sondagem não
        mediu essas técnicas, e começar com a menor redundância mostra cada
        uma como ela é -- se não servir, desce pela repetição."""
        if self.modo_dados == 'auto':
            return contrato.degrau
        if self.modo_dados == '16fsk':
            # Mesmos nomes da escada 'auto', sem o 2-FSK (degrau 3): este
            # cai no rep4, e do 50 baud para baixo ficam os mesmos.
            if contrato.degrau >= R50:
                return 3 + contrato.degrau - R50
            return min(contrato.degrau, 2)
        return 0

    # --- canal aberto: pare-e-espere

    def _abre_canal(self):
        self.aberto = True
        self.confirmado = False
        self.outro_tem_mais = False
        self.sondar_em = None
        self.seq_tx = 0
        if self.pendente_tx is not None:
            # Um pedaço que a sessão anterior mandou e nunca viu confirmado
            # volta para a frente da fila, para recomeçar a sessão não o jogar
            # fora. O preço é que um pedaço entregue cujo ACK se perdeu chega
            # duas vezes; perder é pior.
            self.fila[0:0] = self.pendente_tx[1]
        self.pendente_tx = None
        self.ultimo_rx = 255
        self.degrau_tx = self.degrau_rx = max(self._degrau_inicial(self.contrato_tx),
                                              self._degrau_inicial(self.contrato_rx))
        self.anota(f"velocidade do enlace, a mesma nos dois sentidos: {self.degraus[self.degrau_tx]} "
                   f"(o mais lento entre o que pedi e o que me pediram)")
        self.falhas_rx = 0
        self.sem_progresso = 0
        self.tentativas = 0
        self.progresso_em = self.agora
        self.estado = 'aberto'

    def _turno_dados(self, forcar=False, so_partes=False):
        if self.pendente_tx is None and self.fila:
            pedaco = bytes(self.fila[:DADOS_CARGA])
            del self.fila[:DADOS_CARGA]
            self.pendente_tx = (self.seq_tx, pedaco)
            self.seq_tx = (self.seq_tx + 1) % 255
        if self.pendente_tx is not None:
            seq, pedaco = self.pendente_tx
            d = self.contrato_tx.descritor(self._degrau(), DADOS_N, self.modo_dados)
            carga = bytes([seq, self.ultimo_rx, self._degrau(), len(pedaco)]) + pedaco
            q = self._quadro(d, DADOS, carga)
        else:
            d = self.contrato_tx.descritor(self._degrau(), CTRL_N, self.modo_dados)
            q = self._quadro(d, ACK, bytes([self.ultimo_rx, self._degrau()]))
        return [q]

    def _turno_fim(self):
        d = self.contrato_tx.descritor(self._degrau(), CTRL_N, self.modo_dados)
        return [self._quadro(d, FIM, bytes([self.ultimo_rx]))]

    def _dados(self, quadros, ruins):
        if any(q[0][0] == FIM for q in quadros):
            if self.papel == 'ouvinte':
                # Responde a todo FIM, também a um repetido (o meu se perdeu),
                # e não fala mais nada: não tem relógio próprio para isso.
                # Responde uma vez e não fala mais nada. Se a vez passou, cala
                # sem encerrar: o chamador repete o FIM.
                if self._minha_vez(self._turno_fim(), espera=False):
                    self.anota("FIM do chamador: respondo FIM e calo")
                    self.encerrada = True
            else:
                self.anota("FIM respondido: sessão encerrada")
                self.espera_ate = None
                self.encerrada = True
            return
        if self.papel == 'chamador' and self.fim_tentativas:
            # Já me despedi: só o FIM do outro lado interessa. Um turno ilegível
            # aqui não pode cair no caminho dos dados, ou o ACK no lugar do FIM
            # reabriria a troca.
            return
        responder = not self.confirmado
        progresso = False
        for (tipo, mais, sess, carga), d in quadros:
            if tipo == DADOS:
                seq, ack, pede, n = carga[0], carga[1], carga[2], carga[3]
                pedaco = bytes(carga[4:4 + n])
                # O degrau em que este quadro chegou é o piso do que peço:
                # se o outro lado já desceu, pedir mais rápido seria inútil.
                for k in range(self.ultimo_degrau + 1):
                    if d == self.contrato_rx.descritor(k, d.nbytes, self.modo_dados):
                        self.degrau_rx = max(self.degrau_rx, k)
                if seq == (self.ultimo_rx + 1) % 255 or (self.ultimo_rx == 255 and seq == 0):
                    self.recebido += pedaco
                    self.ultimo_rx = seq
                    self.anota(f"dados seq {seq}: {n} bytes ({d.nome})")
                responder = True
            elif tipo == ACK:
                ack, pede = carga[0], carga[1]
            else:
                continue
            progresso = True
            if self.pendente_tx is not None and ack == self.pendente_tx[0]:
                self.pendente_tx = None
            if pede <= self.ultimo_degrau and pede > self.degrau_tx:
                # Um piso, não uma ordem: quem recebe só sabe das falhas que
                # chegou a ver, e quem transmite pode ter descido sozinho por
                # silêncio. Tomar o pedido ao pé da letra faria os dois
                # oscilarem, um descendo e o outro mandando subir. Subir de
                # novo pede nova sondagem.
                self.anota(f"o outro lado pede o degrau {self.degraus[pede]} para o que envio")
                self.degrau_tx = pede
        if progresso:
            if self.papel == 'chamador':
                # Um turno lido do ouvinte é a resposta, com dados ou só o ACK:
                # desarma o prazo, senão o chamador repetiria o turno com tudo
                # já entregue.
                self.espera_ate = None
            self.progresso_em = self.agora
            self.falhas_rx = 0
            self.sem_progresso = 0
            self.tentativas = 0
            if not self.confirmado:
                self.confirmado = True
                self.anota("CANAL ABERTO: contrato confirmado pelos dois lados")
        elif ruins:
            self.falhas_rx += 1
            self.sem_progresso += 1
            responder = True
            if self.falhas_rx >= 2 and self.degrau_rx < self.ultimo_degrau:
                self.degrau_rx += 1
                self.falhas_rx = 0
                self.anota(f"quadros ilegíveis; pedindo {self.degraus[self.degrau_rx]}")
                if self.degrau_tx < self.degrau_rx:
                    # O pedido vai dentro dos meus quadros, e se eu não leio
                    # os do outro lado é provável que ele também não leia os
                    # meus -- então nunca chega. Desço junto; quem recebe tenta
                    # todos os degraus pelo comprimento, sem precisar ser avisado.
                    self.degrau_tx = self.degrau_rx
                    self.anota(f"desço também o que envio para {self.degraus[self.degrau_tx]}")
            if self.sem_progresso > 8:
                self.anota("canal perdido; voltando ao começo")
                self.reinicia()
                return
        recebeu = any(q[0][0] == DADOS for q in quadros)
        if self.papel == 'ouvinte':
            # O ouvinte responde a todo turno do chamador, na vez dele, e
            # nunca fala sem ser chamado: o prazo, e com ele a repetição, é
            # só do chamador.
            self._minha_vez(self._turno_dados(), espera=False)
            return
        if not progresso and self.espera_ate is not None and self.agora < self.espera_ate:
            # Nada lido, e a vez do ouvinte ainda não acabou: o que soou pode
            # ser o começo da resposta dele. Responder agora seria falar por
            # cima dela. Quando a vez acabar sem resposta lida, o prazo repete
            # o turno.
            return
        self.outro_tem_mais = recebeu
        if self.pendente_tx is not None or self.fila or recebeu:
            responder = True
        if not progresso and ruins:
            # Nada lido e o prazo já passou: é um turno sem resposta, como o
            # do prazo vencido -- conta e desce o degrau do mesmo jeito.
            self.espera_ate = None
            self._sem_resposta()
            return
        if responder:
            # Sem confirmação o outro lado pode não ter o CONTRATO: a resposta
            # é a repetição, que o leva no formato medido.
            self._toca(self._turno_dados() if self.confirmado else self._retransmite(),
                       depois=RESP)
        elif progresso and self.conversa:
            # Conversa: nada a mandar dos dois lados, mas o canal fica aberto.
            # Só o chamador abre trocas, então ele consulta o ouvinte de
            # tempos em tempos -- é na resposta à consulta que o ouvinte manda
            # o que foi digitado lá.
            self.sondar_em = self.agora + CONVERSA_CONSULTA
            self.anota(f"nada a mandar dos dois lados; canal aberto, consulto o outro lado "
                       f"em {CONVERSA_CONSULTA // FS} s")
        elif progresso:
            # O ouvinte respondeu só com ACK: não tem mais nada, e o que ele
            # mandou antes eu já confirmei (senão ele repetiria os DADOS). Do
            # meu lado também acabou. Despedida explícita, em vez de consultar
            # o outro lado para sempre.
            self.anota("tudo entregue nos dois sentidos; mando FIM")
            self.fim_tentativas = 1
            self._toca(self._turno_fim(), depois=RESP)

    # A fila de dados de quem está ocioso: se o canal está aberto, o outro
    # lado calado e há o que mandar, mandar sem ser chamado.
    def ocioso(self):
        return (self.aberto and not self.agenda and self.espera_ate is None
                and self.pendente is None and not self.turno and self.agora >= self.livre_em)
