"""Conversa pelo ar com parâmetros escolhidos à mão, sem negociar nada.

`enlace2.py` acerta volume, velocidade e redundância pelo ar antes do primeiro
byte, e leva minutos nisso. Aqui as duas pontas recebem os mesmos `Params` de
quem as lança, e a estação já nasce em conversa: nada de CHAMA, encontro,
refino, velocidade ou contrato. Sem I/O, como `enlace2.py`; a mesma interface
de estação (`ouvir`/`passo`/`enviar`), para rodar nos mesmos laços de
`saudacao.py` e `movel.py`, com os mesmos quadros (`modula_quadro`/`le_quadro`).

Meio-duplex, sem turnos marcados: quem tem texto espera o ar ficar quieto e
transmite um quadro de 40 bytes, com CRC.

- *Quadro perdido:* com confirmação (o padrão), o receptor responde FICA0/FICA1
  pela paridade do seq. Sem resposta, o emissor repete, até `TENTATIVAS` vezes;
  aí desiste do quadro, diz no diário e segue com o próximo. O receptor descarta
  repetição do mesmo (sessão, seq), mas confirma de novo. Sem confirmação
  (`ack=False`), nada é repetido e um quadro perdido fica perdido.
- *Colisão:* quem está tocando não ouve, então duas pontas podem começar juntas.
  Nenhuma das duas lê o quadro da outra e nenhuma confirma: as duas repetem,
  depois de esperas sorteadas e diferentes por papel. Sem confirmação, os dois
  quadros se perdem.
- *Parâmetros diferentes:* não há como negociar, então a estação diz o que dá
  para ver. O quadro traz um byte de assinatura; a distância entre as marcas de
  abertura e de fechamento dá a duração do quadro do outro. Marca forte com
  quadro ilegível vira um alerta no diário e em `alerta`, com as duas durações.
  `caixa` é local e fica de fora: cada lado tem a sua.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from xfer import crc16
from modem import MARY_TONES, FSK2_TONES
import enlace
from enlace import (FS, Descritor, Parte, MOLDES, LIMIARES, QUADROS, GANHO_FIXO, CAIXA_POS, N_POS,
                    DADOS, DADOS_N, VELOCIDADES, BAUD, LENTO_GUARDA, LENTO_PRE,
                    PRE_SYM, le_quadro, abre, conta_bits)
import enlace2
from enlace2 import (SINAIS, GANHO_SINAL, LIMIAR_SINAL, MARCA_FORTE, MARCA_FRACA, RESP, LAT_CEGO,
                     GUARDA, ATRASO_DET, LAT_OUTRO, RESP_DADOS, LER_CEDO, FECHO_RAZAO, empacota2, modula_quadro,
                     Ouvido2)

VERSAO = 2
CAIXA_INICIAL = enlace2.CAIXA_INICIAL
CARGA = DADOS_N - 5 - 3         # seq, n, assinatura
TENTATIVAS = 6
MARCA_SUBIDA = 1.5              # nível depois da marca / antes dela, para uma marca fraca valer
MARCA_CLARA = 0.30              # abaixo disto uma marca pode ser falsa e não vale alerta
ARRANQUE = 2.0                  # s antes do primeiro quadro: a caixa BT acorda e o volume assenta
# Depois da confirmação: quem a tocou ainda está cego para o que vier (`LAT_CEGO`
# + `GUARDA`). A abertura do quadro seguinte não pode cair nessa cegueira.
FOLGA_ACK = int(1.5 * FS)
ESPERA_MAX = int(15 * FS)       # o ar nunca fica quieto (ventilador?): transmite assim mesmo

# (bits por símbolo, tons de fábrica). As tabelas menores saem da de 16 tons,
# espalhadas pela banda; o 2-FSK é o par `FSK2_TONES`.
MODULACOES = {
    '16fsk': (4, tuple(MARY_TONES)),
    '8fsk': (3, tuple(MARY_TONES[::2])),
    '4fsk': (2, tuple(MARY_TONES[2::4])),
    '2fsk': (1, tuple(FSK2_TONES)),
    'votada': (1, None),
    'multicanal': (5, None),
}
NOMES_MOD = {'16fsk': '16-FSK', '8fsk': '8-FSK', '4fsk': '4-FSK', '2fsk': '2-FSK',
             'votada': '5×2-FSK votada', 'multicanal': '5×2-FSK multicanal'}


def embaralha(seq, dados):
    """XOR com uma sequência fixa, sorteada pelo seq: desfaz a si mesma. Evita
    que um quadro de bytes iguais vire um tom repetido."""
    rng = np.random.RandomState(0x5EED + seq)
    return bytes(a ^ b for a, b in zip(dados, rng.randint(0, 256, len(dados)).astype(np.uint8)))


@dataclass(frozen=True)
class Params:
    """`tom_base` e `passo_tom` são a altura: de onde os tons partem e a
    distância entre eles, em Hz. None mantém a tabela de fábrica. `caixa` é o
    volume, um passo de `CAIXA_POS`, e não entra na assinatura."""
    modulacao: str = '16fsk'
    baud: int = 100
    rep: int = 1
    tom_base: float = None
    passo_tom: float = None
    caixa: int = CAIXA_INICIAL
    ack: bool = True

    @classmethod
    def de_dict(cls, d):
        """Dos formulários e da linha de comando: campo vazio é o de fábrica."""
        c = dict(modulacao=d.get('modulacao') or '16fsk', baud=int(d.get('baud') or 100),
                 rep=int(d.get('rep') or 1), ack=d.get('ack', True) not in (False, 0, '0', 'false'))
        if d.get('tom_base') not in (None, ''):
            c['tom_base'] = float(d['tom_base'])
        if d.get('passo_tom') not in (None, ''):
            c['passo_tom'] = float(d['passo_tom'])
        if d.get('caixa') not in (None, ''):
            c['caixa'] = int(d['caixa'])
        return cls(**c)

    def __post_init__(self):
        if self.modulacao not in MODULACOES:
            raise ValueError(f"modulação desconhecida: {self.modulacao}")
        if not 5 <= int(self.baud) <= 300:
            raise ValueError("baud fora de 5 a 300")
        if not 1 <= int(self.rep) <= 8:
            raise ValueError("rep fora de 1 a 8")
        if not 0 <= int(self.caixa) < N_POS:
            raise ValueError(f"caixa fora de 0 a {N_POS - 1}")
        if self.modulacao in ('votada', 'multicanal') and (self.tom_base or self.passo_tom):
            raise ValueError("as 5×2-FSK têm tons fixos (MFSK_PAIRS): sem tom base nem passo")
        if self.tons and not (300 <= min(self.tons) and max(self.tons) <= 6000):
            raise ValueError(f"tons de {min(self.tons):.0f} a {max(self.tons):.0f} Hz: "
                             "fora de 300 a 6000 Hz")

    @property
    def tons(self):
        bits, fabrica = MODULACOES[self.modulacao]
        if fabrica is None:
            return ()
        if not self.tom_base and not self.passo_tom:
            return fabrica
        base = float(self.tom_base or fabrica[0])
        passo = float(self.passo_tom or (fabrica[-1] - fabrica[0]) / (len(fabrica) - 1))
        return tuple(base + k * passo for k in range(1 << bits))

    @property
    def desc(self):
        bits = MODULACOES[self.modulacao][0]
        baud = int(self.baud)
        # Guarda, preâmbulo e marca da velocidade mais próxima.
        linhas = [(10, LENTO_GUARDA, LENTO_PRE, 'quadro_longo')] + [v[:4] for v in VELOCIDADES]
        _, guarda, pre, marca = min(linhas, key=lambda v: abs(v[0] - baud))
        camada = self.modulacao if self.modulacao in ('votada', 'multicanal') else 'tons'
        return Descritor(self.tons, bits, int(self.rep), DADOS_N, GANHO_FIXO, self.descreve(),
                         baud, guarda, pre, camada=camada, marca=marca)

    def descreve(self):
        t = f"{NOMES_MOD[self.modulacao]} {int(self.baud)} baud rep{int(self.rep)}"
        return t + (f", {self.tons[0]:.0f}-{self.tons[-1]:.0f} Hz"
                    if self.tom_base or self.passo_tom else '')

    def assinatura(self):
        s = f"{self.modulacao}|{int(self.baud)}|{int(self.rep)}|{self.tons}|{self.ack}"
        return crc16(s.encode()) & 0xFF


class _Contrato:
    """Para os painéis, que mostram o contrato de cada sentido."""

    def __init__(self, p):
        self.p = p

    def descreve(self):
        return self.p.descreve()


class Estacao:
    manual = True

    def __init__(self, papel, semente=None, log=None, nome=None, modo_dados='manual', params=None):
        self.papel = papel
        self.nome = nome or papel
        self.params = params or Params()
        self.d = self.params.desc
        self._log = log or (lambda s: None)
        self.rng = np.random.default_rng(semente)
        self.sessao = int(self.rng.integers(256))
        self.ouvido = o = Ouvido2()
        o.moldes = {k: SINAIS[k] for k in ('FICA0', 'FICA1')}
        o.moldes.update({k: MOLDES[k] for k in QUADROS})
        o.limiar = {k: LIMIARES[k] if k in QUADROS else LIMIAR_SINAL for k in o.moldes}
        o.lmax = max(len(m) for m in o.moldes.values())
        self.k = int(self.params.caixa)
        self.fila = bytearray()
        self.recebido = bytearray()
        self.conversa = True
        self.saudado = self.aberto = True
        self.encerrada = False
        self.contrato_tx = self.contrato_rx = _Contrato(self.params)
        self.degraus = (self.params.descreve(),)
        self.pedidos_caixa, self.pedidos_mic, self.pedidos_mic_passo = [], [], []
        self.mic_no_teto = self.mic_no_piso = False
        self.lat_extra = 0
        self.agenda = []
        self.tocando = []
        self.esperando = None
        # Nada sai logo ao abrir: a caixa acorda e o volume assenta antes do primeiro quadro.
        self.livre_em = int(ARRANQUE * FS)
        self.meu_fim = self.quadro_ate = 0
        self.ouvido_em = 0
        self.pendente_tx = None        # dict(x, seq, tent, ate): quadro à espera de confirmação
        self.seq = 0
        self.ult_rx = None
        self.alerta = None
        self.enviados = self.perdidos = self.ilegiveis = 0
        self.bits_errados = self.bits_lidos = 0
        self._lendo = None
        self._no_ar = None
        self._rms = deque(maxlen=int(10 * FS / 2048))
        self._alto_em = -FS
        self._espera_desde = None
        self._t_outra_marca = -60 * FS
        self.anota(f"manual: {self.params.descreve()}, caixa {CAIXA_POS[self.k]:.0%}, "
                   f"assinatura {self.params.assinatura():02x}, quadro de {self.d.segundos:.1f} s, "
                   f"{'com' if self.params.ack else 'sem'} confirmação")

    @property
    def agora(self):
        return self.ouvido.fim

    @property
    def etapa(self):
        return 'dados_tx' if self.fila or self.pendente_tx or self._no_ar else 'dados_rx'

    estado = etapa

    @property
    def confirmado(self):
        return self.pendente_tx is None and not self.fila

    def _degrau(self):
        return 0

    def anota(self, s):
        self._log(f"[{self.nome} {self.agora / FS:7.2f}s] {s}")

    def enviar(self, dados):
        self.fila += bytes(dados)

    def ouvir(self, x):
        # Antes de `acrescenta`: é por isto que se sabe que há som no ar antes
        # de a varredura achar a marca, ~1,6 s depois de ela chegar.
        if len(x) and self.agora >= self.livre_em:
            r = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))
            self._rms.append(r)
            piso = np.percentile(self._rms, 20)
            if len(self._rms) > 20 and r > max(4 * piso, 0.002):
                self._alto_em = self.agora + len(x)
        self.ouvido.acrescenta(x)

    def desloca_saida(self, n):
        if n <= 0:
            return
        if self.ouvido.cegos:
            a, b = self.ouvido.cegos[-1]
            self.ouvido.cegos[-1] = (a + n, b + n)
        self.livre_em += n
        self.meu_fim += n
        if self.pendente_tx is not None:
            self.pendente_tx['ate'] += n
        if self._no_ar is not None:
            a, m, r = self._no_ar
            self._no_ar = (a + n, m, r)

    def _regua(self):
        """Os tons em uso, escolhidos à mão: a mesma régua, aqui sem medida."""
        tons = self.params.tons
        return [dict(rotulo='Em uso', valor=f"{len(tons)} tons" if tons else 'Fixos da modulação',
                     regua=[dict(hz=float(f), estado='usado') for f in tons])]

    def percurso(self, mic=None, parada=False):
        """O mesmo formato de `enlace2.Estacao.percurso`: aqui, duas etapas."""
        t = self.agora
        if self._no_ar is not None and t < self._no_ar[0] + self._no_ar[1] + LAT_CEGO:
            andamento = dict(sentido='enviando', s=round((self._no_ar[0] + self._no_ar[1] + LAT_CEGO - t) / FS))
        elif self.pendente_tx is not None and self._lendo is None:
            andamento = dict(sentido='ouvindo', s=max(0, round((self.pendente_tx['ate'] - t) / FS)))
        else:
            andamento = dict(sentido='ouvindo', s=None)
        conversa = dict(id='conversa', nome='Conversa', situacao='feita' if parada else 'agora', valores=[
            dict(rotulo='Quadros enviados', valor=str(self.enviados), lado='envio'),
            dict(rotulo='Perdidos', valor=str(self.perdidos)),
            dict(rotulo='Ilegíveis', valor=str(self.ilegiveis))])
        if self.bits_lidos:
            conversa['valores'].append(dict(
                rotulo='Bits corrigidos',
                valor=f"{self.bits_errados} de {self.bits_lidos} ({self.bits_errados / self.bits_lidos:.1%})"))
        if not parada and self.enviados and not self.fila and self.pendente_tx is None:
            # Tudo entregue, e o canal segue aberto: a tela diz, para não parecer travado.
            conversa['valores'] += [dict(rotulo='A enviar', valor='nada', lado='envio'),
                                    dict(rotulo='Canal', valor='aberto')]
        if not parada:
            conversa['andamento'] = andamento
        return [
            dict(id='parametros', nome='Parâmetros', situacao='feita', valores=[
                dict(rotulo='Modo', valor=self.params.descreve().replace(' rep', ', repetição ')),
                dict(rotulo='Volume', valor=f"{CAIXA_POS[self.k]:.0%}", lado='envio'),
                dict(rotulo='Microfone', valor='—' if mic is None else f"{mic:.0%}")]),
            dict(id='tons', nome='Tons', situacao='feita', valores=self._regua()),
            conversa,
        ]

    def atividade(self):
        t = self.agora
        base = f"manual, {self.params.descreve()}"
        if self._no_ar is not None and t < self._no_ar[0] + self._no_ar[1] + LAT_CEGO:
            return f"{base}: tocando {self._no_ar[2]}"
        if self._lendo is not None:
            return f"{base}: lendo um quadro (marca há {(t - self._lendo['p']) / FS:.0f} s)"
        if self.pendente_tx is not None:
            return f"{base}: esperando confirmação ({max(0, self.pendente_tx['ate'] - t) / FS:.0f} s)"
        return f"{base}: escutando"

    # --- transmissão

    def _parte(self, x, rotulo):
        x = np.asarray(x, dtype=np.float64).view(Parte)
        x.rotulo, x.caixa_marcas = rotulo, [(0, self.k)]
        return x

    def _quadro(self):
        pedaco = bytes(self.fila[:CARGA])
        del self.fila[:len(pedaco)]
        cabeca = bytes([self.seq & 0xFF, len(pedaco), self.params.assinatura()])
        corpo = empacota2(DADOS, self.sessao, cabeca + embaralha(self.seq & 0xFF, pedaco), DADOS_N)
        seq = self.seq
        self.seq = (self.seq + 1) & 0xFF
        return dict(seq=seq, n=len(pedaco), tent=0,
                    x=self._parte(modula_quadro(self.d, corpo), f"quadro seq {seq}"))

    def _ate(self, a, n, tent=1):
        """Até quando esperar a confirmação; a folga sorteada, e maior no
        ouvinte, é o que separa as duas pontas depois de uma colisão. Da segunda
        tentativa em diante ela cresce com o quadro: uma folga menor que ele não
        tira as repetições de cima uma da outra."""
        escala = n / FS if tent > 1 else 0.5
        folga = self.rng.uniform(0, escala if self.papel == 'chamador' else 3 * escala)
        return int(a + n + LAT_CEGO + RESP_DADOS + ATRASO_DET + LAT_OUTRO + len(SINAIS['FICA0'])
                   + ATRASO_DET + 0.2 * FS + folga * FS)

    def _toca(self, x):
        a, n = self.agora, len(x)
        self.livre_em = self.meu_fim = a + n + LAT_CEGO
        self.ouvido.cego(a, a + n + LAT_CEGO + GUARDA)
        self._no_ar = (a, n, x.rotulo)
        self.tocando = [x.rotulo]
        self._lendo = None
        self.anota(f"mando {x.rotulo} (caixa {CAIXA_POS[self.k]:.0%}) em {a / FS:.1f}s")
        return x

    def _ar_ocupado(self):
        return self.agora - self._alto_em < int(0.3 * FS)

    def _transmite(self):
        if self.agora < self.livre_em:
            return None
        if self.agenda and self.agenda[0][0] <= self.agora:
            return self._toca(self.agenda.pop(0)[1])
        if self.agenda or self._lendo is not None:
            return None
        p = self.pendente_tx
        if p is not None and self.agora < p['ate']:
            return None
        if p is None and not self.fila:
            return None
        if self._ar_ocupado():
            if self._espera_desde is None:
                self._espera_desde = self.agora
            if self.agora - self._espera_desde < ESPERA_MAX:
                return None
        self._espera_desde = None
        if p is not None and p['tent'] >= TENTATIVAS:
            self.perdidos += 1
            self.anota(f"ATENÇÃO: quadro seq {p['seq']} sem confirmação em {TENTATIVAS} tentativas; "
                       f"desisto dele ({p['n']} bytes perdidos)")
            self.pendente_tx = p = None
            if not self.fila:
                return None
        if p is None:
            p = self._quadro()
            self.enviados += 1
        p['tent'] += 1
        if p['tent'] > 1:
            self.anota(f"sem confirmação do seq {p['seq']}: tentativa {p['tent']}")
        if self.params.ack:
            p['ate'] = self._ate(self.agora, len(p['x']), p['tent'])
            self.pendente_tx = p
        return self._toca(p['x'])

    # --- recepção

    def _e_fechamento(self, L):
        """A marca lida é a de fechamento de um quadro cuja abertura se perdeu?
        Como em `enlace2`: antes dela há corpo e depois dela o som parou. Sem isso a
        leitura fantasma, de um quadro inteiro, engoliria a abertura da repetição.
        Quadro curto demais não dá para distinguir: seu fim também vem seguido de
        silêncio. None: falta áudio."""
        p, lm = L['p'], self.d.lm
        ini, fim = p + lm + int(0.3 * FS), p + lm + int(1.8 * FS)
        if self.d.amostras < lm + int(2.5 * FS):
            return False
        if self.ouvido.varrido < fim:
            return None
        rms = lambda x: float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
        antes = rms(self.ouvido.trecho(p - int(1.7 * FS), p - int(0.3 * FS)))
        marca = rms(self.ouvido.trecho(p, p + lm))
        depois = rms(self.ouvido.trecho(ini, fim))
        return depois < FECHO_RAZAO * antes and depois < FECHO_RAZAO * marca

    def _le(self):
        L = self._lendo
        if L is None:
            return
        p, d = L['p'], self.d
        if L['sc'] < MARCA_CLARA and 'viva' not in L:
            # No ocioso o ruído pode imitar a marca e prender a estação. Depois de um começo
            # de quadro de verdade o nível sobe; sem isso, larga.
            ate = p + d.lm + int(0.6 * FS)
            if self.ouvido.fim < ate:
                return
            rms = lambda x: float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
            if rms(self.ouvido.trecho(p + d.lm + int(0.1 * FS), ate)) < MARCA_SUBIDA * rms(
                    self.ouvido.trecho(p - int(0.6 * FS), p - int(0.1 * FS))):
                self._lendo = None
                return
            L['viva'] = True
        if 'fechamento' not in L:
            f = self._e_fechamento(L)
            if f is not None:
                L['fechamento'] = f
            if f:
                self._lendo = None
                self.quadro_ate = p + d.lm
                self.anota(f"marca em {p / FS:.2f}s é fechamento (a abertura se perdeu): "
                           "largo a leitura e espero a repetição")
                return
        if 'cedo' not in L and self.ouvido.fim >= p + d.amostras + LER_CEDO:
            # Sem esperar a varredura achar a marca de fechamento; se falhar, a
            # leitura de sempre vem depois, com ela.
            L['cedo'] = True
            corpo, llr = le_quadro(self.ouvido.trecho(p, p + d.amostras + LER_CEDO), 0, d, None)
            if corpo is not None:
                self._lendo = None
                self.quadro_ate = p + d.amostras
                self.ilegiveis = 0
                self.alerta = None
                self.ouvido_em = self.agora
                self._recebeu(corpo, llr)
                return
        if self.ouvido.varrido < p + d.amostras + FS // 2:
            return
        self._lendo = None
        alvo = p + d.amostras - d.lm
        fecho = [m for m in self.ouvido.marcas
                 if m[1] == d.marca and abs(m[0] - alvo) <= int(0.004 * d.amostras)]
        audio = self.ouvido.trecho(p, p + d.amostras + FS)
        corpo, llr = le_quadro(audio, 0, d, (fecho[0][0] - p) if fecho else None)
        if corpo is None and fecho:
            corpo, llr = le_quadro(audio, 0, d, None)
        self.quadro_ate = p + d.amostras
        if corpo is None:
            self._ilegivel(L, bool(fecho))
            return
        self.ilegiveis = 0
        self.alerta = None
        self.ouvido_em = self.agora
        self._recebeu(corpo, llr)

    def _recebeu(self, corpo, llr=None):
        tipo, _, sessao, carga = abre(corpo)
        if tipo != DADOS:
            return
        seq, n, ass = carga[0], carga[1], carga[2]
        c = conta_bits(corpo, llr, self.d)
        if c:
            self.bits_errados += c[0]
            self.bits_lidos += c[1]
        bits = f", {c[0]} de {c[1]} bits corrigidos ({c[0] / c[1]:.1%})" if c else ""
        if ass != self.params.assinatura():
            self.alerta = (f"o outro lado assina {ass:02x} e este {self.params.assinatura():02x}: "
                           "parâmetros diferentes, ainda que este quadro tenha passado")
            self.anota(f"ATENÇÃO: {self.alerta}")
        if (sessao, seq) == self.ult_rx:
            self.anota(f"repetição do seq {seq}; confirmo de novo{bits}")
        else:
            self.ult_rx = (sessao, seq)
            self.recebido += embaralha(seq, carga[3:3 + n])
            self.anota(f"li quadro seq {seq}: {n} bytes (total {len(self.recebido)}){bits}")
        if self.params.ack:
            x = SINAIS[f"FICA{seq & 1}"] * GANHO_SINAL
            self.agenda.append((max(self.agora, self.quadro_ate + RESP_DADOS),
                                self._parte(x, f"confirmação do seq {seq}")))

    def _ilegivel(self, L, com_fecho):
        self.ilegiveis += 1
        if L['sc'] < MARCA_CLARA and not com_fecho:
            self.anota(f"marca fraca ({L['sc']:.2f}) em {L['p'] / FS:.2f}s sem quadro legível")
            return
        d = self.d
        seguinte = [m[0] for m in self.ouvido.marcas
                    if m[1] == L['marca'] and m[2] >= MARCA_CLARA and m[0] > L['p'] + d.lm
                    and m[0] < L['p'] + d.amostras + FS // 2]
        medida = (seguinte[0] - L['p'] + d.lm) / FS if seguinte else None
        minha = d.segundos
        if seguinte and not com_fecho and self.ilegiveis < 2:
            # Pode ter sido o fechamento do quadro anterior lido como abertura:
            # a próxima marca forte, já na memória, é o começo de verdade. Um
            # só ainda não é parâmetro diferente.
            self.anota(f"quadro ilegível (marca {L['sc']:.2f} em {L['p'] / FS:.2f}s); "
                       f"releio da marca em {seguinte[0] / FS:.2f}s")
            self.quadro_ate = seguinte[0]
            self._lendo = dict(p=seguinte[0], marca=L['marca'], sc=MARCA_CLARA)
            return
        if com_fecho:
            motivo = (f"a duração bate ({minha:.1f} s) e não li: tons ou modulação diferentes, "
                      "ou canal ruim")
        elif medida is not None and abs(medida - minha) > 0.01 * minha:
            motivo = (f"o quadro de lá mede {medida:.1f} s e o daqui {minha:.1f} s: "
                      "baud, rep ou modulação diferentes")
        else:
            motivo = (f"sem marca de fechamento em {minha:.1f} s: quadro de lá mais longo "
                      "(parâmetros diferentes), ou canal ruim")
        self.alerta = motivo
        self.anota(f"ATENÇÃO: quadro ilegível (marca {L['sc']:.2f} em {L['p'] / FS:.2f}s): {motivo}")

    def _marca(self, pos, tipo, sc):
        if tipo != self.d.marca:
            if sc >= MARCA_CLARA and self.agora - self._t_outra_marca > 30 * FS:
                self._t_outra_marca = self.agora
                self.alerta = ("o outro lado usa a marca de outra faixa de velocidade: "
                               "baud diferente")
                self.anota(f"ATENÇÃO: marca {tipo} ({sc:.2f}): {self.alerta}")
            return
        L = self._lendo
        if L is None:
            self._lendo = dict(p=pos, marca=tipo, sc=sc)
            self.anota(f"marca de quadro ({tipo}, {sc:.2f}) em {pos / FS:.2f}s: lendo")
        elif (pos > L['p'] and sc >= MARCA_FORTE and L['sc'] < MARCA_FRACA
              and abs(pos - (L['p'] + self.d.amostras - self.d.lm)) > int(0.004 * self.d.amostras)):
            # Como em `enlace2`: uma falsa não segura a leitura da verdadeira.
            self.anota(f"marca fraca em {L['p'] / FS:.2f}s ({L['sc']:.2f}) largada pela forte "
                       f"em {pos / FS:.2f}s")
            self._lendo = dict(p=pos, marca=tipo, sc=sc)

    def _sinal(self, pos, tipo, sc):
        p = self.pendente_tx
        if p is not None and tipo == f"FICA{p['seq'] & 1}" and pos >= self.meu_fim:
            self.anota(f"confirmado o seq {p['seq']} ({sc:.2f}), {p['tent']}ª tentativa")
            self.pendente_tx = None
            self.ouvido_em = self.agora
            self.livre_em = max(self.livre_em, self.agora + FOLGA_ACK)

    def passo(self):
        for pos, tipo, sc in self.ouvido.varre():
            if pos < self.meu_fim:
                continue
            if tipo in SINAIS:
                self._sinal(pos, tipo, sc)
            elif pos >= self.quadro_ate:
                self._marca(pos, tipo, sc)
        self._le()
        return self._transmite()
