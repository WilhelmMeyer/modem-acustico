"""Volume do sistema: a caixa e o microfone desta máquina, na posição do
controle deslizante (0 a 1, o "%" que o usuário vê no sistema).

Runtime, como `saudacao.py`: `enlace.py` só diz qual passo quer
(`Parte.caixa_marcas`, `Estacao.pedidos_mic`), e é aqui que o passo vira
volume de verdade. Ganho digital não tira um alto-falante do limitador; o
volume do sistema tira.

As duas pontas usam a mesma escada de posições, e não uma tabela de dB por
máquina, para partirem do mesmo ponto do controle.

Linux: `wpctl`, no sink e na fonte padrão; o número dele é o % do GNOME.

Windows: um `powershell` só, vivo a sessão inteira, com a interface
IAudioEndpointVolume compilada uma vez, lendo `get spk|mic` e
`set spk|mic <posição>` linha a linha; a posição é o
`MasterVolumeLevelScalar`, o % do controle do Windows. Abrir um powershell por
troca leva segundos, e a troca tem de cair num silêncio de 0,6 s.

Os volumes de partida ficam num arquivo na pasta do usuário enquanto a
sessão roda; `restaura` os devolve no fim, e quem começar e achar o arquivo
ainda lá (a sessão anterior morreu sem restaurar) restaura dele primeiro.
"""

import json
import re
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

# Passos da caixa, posição do controle, iguais nas duas pontas, índice absoluto:
# a mesma tabela que `enlace.CAIXA_POS` (o comprimento tem de bater; `saudacao`
# confere). A chamada sobe do passo 1 (20%) até ser respondida, e a sondagem e o
# contrato andam em volta do passo travado.
# `AUDIOFSK_CAIXA="0.15,0.2,0.3,0.4,0.55,0.7,1"` troca a tabela sem mexer no código.
CAIXA = (0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 1.00)
# Microfone de quem escuta, relativo ao de partida: o passo 1 é 0,6 dele.
MIC_PASSOS = (1.0, 0.6)
# Ao acertar o som, quem ouve fraco sobe o próprio microfone por estas
# posições do controle, a partir da primeira acima da de partida, até 100%.
# O microfone fica onde parou; a volta da sondagem (MIC_PASSOS) passa a ser
# em torno dele.
MIC_POS = (0.20, 0.30, 0.45, 0.60, 0.75, 1.00)
ARQUIVO = Path.home() / '.audiofsk-volume-original.json'

_PS = r'''
Add-Type -TypeDefinition @"
using System; using System.Runtime.InteropServices;
[Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioEndpointVolume {
 int f(); int g(); int h();
 int SetMasterVolumeLevel(float l, Guid c); int SetMasterVolumeLevelScalar(float l, Guid c);
 int GetMasterVolumeLevel(out float l); int GetMasterVolumeLevelScalar(out float l);
 int a(); int b(); int c(); int d(); int SetMute(bool m, Guid c); int GetMute(out bool m);
 int j(); int k(); int l(); int m(); int GetVolumeRange(out float mn, out float mx, out float st);
}
[StructLayout(LayoutKind.Sequential)] public struct PK { public Guid g; public int id; }
[StructLayout(LayoutKind.Sequential)] public struct PV { public short vt, r1, r2, r3; public IntPtr p, p2; }
[Guid("886d8eeb-8cf2-4446-8d02-cdba1dbdcf99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IPropertyStore { int a(); int b(); int GetValue(ref PK k, out PV v); }
[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDevice { int Activate(ref Guid id, int ctx, IntPtr p, [MarshalAs(UnmanagedType.IUnknown)] out object o);
 int OpenPropertyStore(int acc, out IPropertyStore s); }
[Guid("0BD7A1BE-7A1A-44DB-8397-CC5392387B5E"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceCollection { int GetCount(out int n); int Item(int i, out IMMDevice d); }
[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceEnumerator { int EnumAudioEndpoints(int flow, int state, out IMMDeviceCollection c);
 int GetDefaultAudioEndpoint(int flow, int role, out IMMDevice d); }
[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")] class MMDeviceEnumerator {}
public class Vol {
 public static string[] Nome = new string[2];
 static string Amigavel(IMMDevice d) {
  IPropertyStore s; d.OpenPropertyStore(0, out s);
  var k = new PK { g = new Guid("a45c254e-df1c-4efd-8020-67d146a850e0"), id = 14 }; PV v; s.GetValue(ref k, out v);
  return v.vt == 31 ? Marshal.PtrToStringUni(v.p) : ""; }
 static IMMDevice Acha(int flow) {
  var en = (IMMDeviceEnumerator)(new MMDeviceEnumerator()); IMMDevice d;
  string n = Nome[flow];
  if (!String.IsNullOrEmpty(n)) {
   IMMDeviceCollection c; en.EnumAudioEndpoints(flow, 1, out c); int k; c.GetCount(out k);
   for (int i = 0; i < k; i++) { c.Item(i, out d); string a = Amigavel(d);
    if (a == n || a.Contains(n) || (n.Length >= 20 && a.StartsWith(n))) return d; }
   throw new Exception("nenhum dispositivo ativo com o nome " + n); }
  en.GetDefaultAudioEndpoint(flow, 0, out d); return d; }
 public static string Qual(int flow) { return Amigavel(Acha(flow)); }
 static IAudioEndpointVolume V(int flow) {
  IMMDevice d = Acha(flow);
  var iid = typeof(IAudioEndpointVolume).GUID; object o; d.Activate(ref iid, 23, IntPtr.Zero, out o); return (IAudioEndpointVolume)o; }
 public static string Get(int flow) { var v=V(flow); float s; bool m; v.GetMasterVolumeLevelScalar(out s); v.GetMute(out m);
  return String.Format(System.Globalization.CultureInfo.InvariantCulture, "ok {0:F3} {1}", s, m ? 1 : 0); }
 public static string Set(int flow, float s) { var v=V(flow);
  v.SetMasterVolumeLevelScalar(Math.Max(0f, Math.Min(1f, s)), Guid.Empty); v.SetMute(false, Guid.Empty); return Get(flow); }
}
"@
[Console]::Out.WriteLine("pronto"); [Console]::Out.Flush()
while ($null -ne ($l = [Console]::In.ReadLine())) {
 $p = $l.Split(' '); $flow = if ($p[1] -eq 'mic') { 1 } else { 0 }
 try {
  if ($p[0] -eq 'nome') { [Vol]::Nome[$flow] = ($p[2..($p.Length-1)] -join ' '); $r = "ok 1 " + [Vol]::Qual($flow) }
  elseif ($p[0] -eq 'set') { $r = [Vol]::Set($flow, [float]::Parse($p[2], [System.Globalization.CultureInfo]::InvariantCulture)) }
  else { $r = [Vol]::Get($flow) }
 } catch { $r = "erro " + $_.Exception.Message }
 [Console]::Out.WriteLine($r); [Console]::Out.Flush()
}
'''


def _tabela_caixa():
    t = os.environ.get('AUDIOFSK_CAIXA')
    if t:
        return tuple(float(x) for x in t.split(','))
    return CAIXA


class Volume:
    """`le(qual)` e `poe(qual, v)`, v a posição do controle (0 a 1), com qual 'spk' ou 'mic'; `caixa(k)` e
    `mic(k)` põem o passo k. Uma instância por processo, chamada de uma
    thread só por vez (a trava garante)."""

    def __init__(self, spk=None, mic=None):
        """`spk`/`mic`: nome (ou parte) do dispositivo que a sessão usa. No
        Windows o volume vai nesse dispositivo e não no padrão do sistema, que muda
        sozinho quando se pluga um adaptador USB. No Linux o wpctl segue o padrão."""
        self.trava = threading.Lock()
        self.nomes = {}
        self._ids = {}
        if sys.platform != 'win32':
            for qual, nome in (('spk', spk), ('mic', mic)):
                if nome:
                    self._ids[qual], self.nomes[qual] = self._id_wpctl(qual, nome)
        self.caixa_pos = _tabela_caixa()
        self._ps = None
        if sys.platform == 'win32':
            fd, nome = tempfile.mkstemp(suffix='.ps1', prefix='audiofsk-vol-')
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(_PS)
            self._ps = subprocess.Popen(
                ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', nome],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding='utf-8', errors='replace', bufsize=1,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            pronto = self._ps.stdout.readline().strip()
            if pronto != 'pronto':
                raise RuntimeError(f"powershell de volume não respondeu: {pronto!r}")
            for qual, nome in (('spk', spk), ('mic', mic)):
                if nome:
                    self._ps.stdin.write(f'nome {qual} {nome}\n')
                    self._ps.stdin.flush()
                    r = self._ps.stdout.readline().strip()
                    if not r.startswith('ok '):
                        raise RuntimeError(f"volume ({qual} {nome!r}): {r}")
                    self.nomes[qual] = r.split(' ', 2)[2]
        self.original = None
        self.mic_base = None

    # --- baixo nível

    def _ps_cmd(self, linha):
        self._ps.stdin.write(linha + '\n')
        self._ps.stdin.flush()
        r = self._ps.stdout.readline().strip()
        if not r.startswith('ok '):
            raise RuntimeError(f"volume ({linha}): {r}")
        return float(r.split()[1])

    def _alvo(self, qual):
        if qual in self._ids:
            return self._ids[qual]
        return '@DEFAULT_AUDIO_SINK@' if qual == 'spk' else '@DEFAULT_AUDIO_SOURCE@'

    @staticmethod
    def _id_wpctl(qual, nome):
        """O id do nó do PipeWire cujo nome contém `nome`, entre as saídas
        (spk) ou entradas (mic) de `wpctl status -n`."""
        r = subprocess.run(['wpctl', 'status', '-n'], capture_output=True, text=True,
                           timeout=5).stdout
        secao = 'Sinks:' if qual == 'spk' else 'Sources:'
        dentro = False
        for l in r.splitlines():
            if secao in l:
                dentro = True
                continue
            if dentro and l.strip(' │├└─*') == '':
                dentro = False
            m = re.match(r'[\s│├└─*]*(\d+)\.\s+(\S+)', l) if dentro else None
            # o PortAudio diz bluez_output.41:42:..., o PipeWire 41_42_...
            if m and nome.replace(':', '_') in m.group(2):
                return m.group(1), m.group(2)
        raise RuntimeError(f"nenhum nó do PipeWire com o nome {nome!r} em {secao}")

    def le(self, qual):
        with self.trava:
            if self._ps is not None:
                return self._ps_cmd(f'get {qual}')
            r = subprocess.run(['wpctl', 'get-volume', self._alvo(qual)],
                               capture_output=True, text=True, timeout=5).stdout
            return float(r.split()[1])

    def poe(self, qual, v):
        """Põe o volume na posição `v` e devolve o que ficou. O microfone sai
        sempre do mudo."""
        with self.trava:
            if self._ps is not None:
                return self._ps_cmd(f'set {qual} {v:.3f}')
            v = max(0.0, min(1.0, v))
            subprocess.run(['wpctl', 'set-volume', self._alvo(qual), f'{v:.3f}'],
                           check=True, timeout=5)
            if qual == 'mic':
                subprocess.run(['wpctl', 'set-mute', self._alvo(qual), '0'], timeout=5)
            return v

    # --- passos

    def caixa(self, k):
        return self.poe('spk', self.caixa_pos[k])

    def mic(self, k):
        return self.poe('mic', self.mic_base * MIC_PASSOS[k])

    def proximo_mic(self, direcao=1):
        """A posição de MIC_POS um passo acima (`direcao` > 0) ou abaixo do
        microfone em uso, ou None no teto/piso. Só calcula: quem move é
        `Agenda` com 'micpos'."""
        if direcao > 0:
            acima = [v for v in MIC_POS if v > self.mic_base + 0.005]
            return acima[0] if acima else None
        abaixo = [v for v in MIC_POS if v < self.mic_base - 0.005]
        return abaixo[-1] if abaixo else None

    # --- partida e fim

    def guarda_original(self, log=print):
        """Restaura o que uma sessão anterior deixou sem restaurar, lê os
        volumes de partida e os guarda em ARQUIVO."""
        if ARQUIVO.exists():
            try:
                o = json.loads(ARQUIVO.read_text())
                self.poe('spk', o['spk'])
                self.poe('mic', o['mic'])
                log(f"[volume] a sessão anterior não restaurou o volume; restaurado do arquivo: "
                    f"caixa {o['spk']:.0%}, microfone {o['mic']:.0%}")
            except Exception as e:
                log(f"[volume] arquivo de volume antigo ilegível ({e}); ignorado")
        self.original = {'spk': self.le('spk'), 'mic': self.le('mic')}
        self.mic_base = self.original['mic']
        ARQUIVO.write_text(json.dumps(dict(self.original, pid=os.getpid())))
        log(f"[volume] de partida: caixa {self.original['spk']:.0%}, microfone "
            f"{self.original['mic']:.0%}; passos da caixa "
            f"{', '.join(f'{v:.0%}' for v in self.caixa_pos)}, microfone "
            f"{', '.join(f'{self.mic_base * m:.0%}' for m in MIC_PASSOS)}")

    def restaura(self, log=print):
        if self.original is None:
            return
        try:
            self.poe('spk', self.original['spk'])
            self.poe('mic', self.original['mic'])
            ARQUIVO.unlink(missing_ok=True)
            log(f"[volume] restaurado: caixa {self.original['spk']:.0%}, "
                f"microfone {self.original['mic']:.0%}")
        except Exception as e:
            log(f"[volume] ERRO ao restaurar ({e}); o arquivo {ARQUIVO} fica para a próxima sessão")

    def fecha(self):
        if self._ps is not None:
            try:
                self._ps.stdin.close()
                self._ps.wait(timeout=3)
            except Exception:
                self._ps.kill()


class Agenda(threading.Thread):
    """Trocas de volume na hora certa do relógio de parede, numa thread à
    parte (o `wpctl` leva ~10 ms, e o laço principal não pode esperar).
    `avisos` recebe uma linha por troca, para o diário."""

    def __init__(self, vol):
        super().__init__(daemon=True)
        self.vol = vol
        self.cv = threading.Condition()
        self.fila = []                 # [(t, qual, passo)]
        self.avisos = []
        self.atual = {'spk': None, 'mic': None}
        self.parar = False

    def poe(self, t, qual, passo):
        with self.cv:
            self.fila.append((t, qual, passo))
            self.fila.sort(key=lambda x: x[0])
            self.cv.notify()

    def limpa(self, qual):
        with self.cv:
            self.fila = [x for x in self.fila if x[1] != qual]

    def fecha(self):
        with self.cv:
            self.parar = True
            self.cv.notify()
        self.join(timeout=5)

    def run(self):
        import time
        while True:
            with self.cv:
                while not self.parar and (not self.fila or self.fila[0][0] > time.time()):
                    self.cv.wait(None if not self.fila else max(0.0, self.fila[0][0] - time.time()))
                if self.parar:
                    return
                t, qual, passo = self.fila.pop(0)
            if qual == 'micpos':
                # Microfone a uma posição nova, que passa a ser a base dos
                # passos relativos (`mic_base` já foi posto por quem pediu).
                try:
                    v = self.vol.poe('mic', passo)
                    self.atual['mic'] = 0
                    self.avisos.append(f"[volume] microfone em {v:.0%} "
                                       f"({time.time() - t:+.2f}s da hora marcada)")
                except Exception as e:
                    self.avisos.append(f"[volume] ERRO ao pôr o microfone em {passo:.0%}: {e}")
                continue
            if self.atual[qual] == passo:
                continue
            try:
                v = self.vol.caixa(passo) if qual == 'spk' else self.vol.mic(passo)
                self.atual[qual] = passo
                nome = 'caixa' if qual == 'spk' else 'microfone'
                self.avisos.append(f"[volume] {nome} no passo {passo}: {v:.0%} "
                                   f"({time.time() - t:+.2f}s da hora marcada)")
            except Exception as e:
                self.avisos.append(f"[volume] ERRO ao pôr {qual} no passo {passo}: {e}")
