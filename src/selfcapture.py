"""Record this machine transmitting to itself, through the air, with no second
computer and no serial cable.

Speaker out, microphone in, one soundcard. The air, the room's comb response,
the output limiter and the microphone are all real. What this path cannot
reproduce:

  * With a wired speaker both directions share the soundcard's clock, so
    there is no sample-rate drift between transmitter and receiver, and timing
    results come out optimistic. A *Bluetooth* speaker has its own crystal and
    puts the drift back, at the cost of a lossy codec; `--link` records which
    one was used, and the two must not be averaged together.
  * One room, one pair of transducers. A far machine's speaker and microphone
    have their own response.

Output is the `recording.py` pair: a WAV plus a JSON sidecar.

    ./venv/bin/python selfcapture.py --mode mary --fec --gain 0.5 --trials 3
"""

import argparse
import queue
import sys
import time

import numpy as np
import sounddevice as sd

import aviso
import fec
import recording
from modem import (FSKModulator, MFSKModulator, MFSK_PAIRS, MaryModulator,
                   MARY_BITS, chirp, SYNC_CHIRP, GRAY_ATUAL)

FS = 48000
BLOCK = 2048
MFSK_BAUD = 100
PREAMBLE = bytes([0x55] * 10 + [0xFF])

# Alphabet the random payload is drawn from.
ALPHABET = (b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            b"abcdefghijklmnopqrstuvwxyz0123456789")


def make_text_payload(seed, n):
    import random
    rng = random.Random(seed)
    return bytes(rng.choice(ALPHABET) for _ in range(n))


class Recorder:
    """Input through the callback API: some host APIs on Windows have no
    blocking read."""

    def __init__(self, device):
        self.q = queue.Queue()
        self.stream = sd.InputStream(samplerate=FS, channels=1,
                                     blocksize=BLOCK, device=device,
                                     callback=self._cb)

    def _cb(self, indata, frames, time_info, status):
        self.q.put(indata[:, 0].copy())

    def __enter__(self):
        self.stream.start()
        time.sleep(0.3)
        self.drain()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()

    def drain(self):
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def collect(self, secs):
        chunks = []
        t0 = time.time()
        while time.time() - t0 < secs:
            try:
                chunks.append(self.q.get(timeout=0.5))
            except queue.Empty:
                pass
        while True:
            try:
                chunks.append(self.q.get_nowait())
            except queue.Empty:
                break
        return np.concatenate(chunks) if chunks else np.zeros(0)


def build_modulator(args):
    if args.mode == 'fsk':
        return FSKModulator(fs=FS, baud=1200)
    if args.mode == 'mary':
        return MaryModulator(fs=FS, baud=MFSK_BAUD, gap=args.gap or 0.0,
                             chord=args.chord)
    return MFSKModulator(fs=FS, baud=MFSK_BAUD, grouped=args.grouped)


def build_burst(args, mod, payload):
    """The samples this burst puts on the air, gain applied.

    The preamble comes from `fec.preamble_bits`, so it cannot drift from what the
    receiver expects.
    """
    k = len(MFSK_PAIRS)
    if args.fec:
        if args.mode == 'fsk':
            sys.exit("[selfcapture] --fec so em mfsk ou mary")
        if args.mode == 'mary':
            bits = fec.frame(payload, repeat=args.repeat)
            pre = fec.preamble_bits('mary', symbol_bits=MARY_BITS)
            tail = mod.idle(6)
        else:
            bits = (fec.frame_parallel(payload, k, repeat=args.repeat)
                    if args.parallel else fec.frame(payload, repeat=args.repeat))
            pre = fec.preamble_bits(args.mode, npairs=k, parallel=args.parallel)
            tail = mod.idle(4)
        samples = np.concatenate([mod.modulate_bits(pre),
                                  mod.modulate_bits(list(bits)), tail])
    else:
        samples = mod.modulate(PREAMBLE + payload)
        # Bell 202 needs no tail and offers none; the other two strand their
        # last symbol without one.
        if args.mode != 'fsk':
            samples = np.concatenate([samples, mod.idle(4)])
    samples = samples * args.gain
    args.sync_span_symbols = 0.0
    if args.sync_chirp:
        # The sweep goes out at the same amplitude as the data, so a limiter treats
        # both alike. The silence after it separates the sweep's own decay from the
        # first symbol, so the matched filter's peak is not sitting on top of data.
        lead = chirp(FS, *SYNC_CHIRP) * args.gain
        hush = np.zeros(int(args.sync_hush * FS))
        body = np.concatenate([hush, samples, hush])
        # The span between the two detections, in symbols, as transmitted. The
        # receiver divides the interval it *measures* by this to get the real
        # samples per symbol. Expressed in symbols rather than samples so the
        # silences scale with the clock like everything else.
        args.sync_span_symbols = len(body) / (FS / MFSK_BAUD)
        samples = np.concatenate([lead, body, lead])
    return samples.astype(np.float32)


def build_chirp(f0, f1, secs, gain):
    n = int(secs * FS)
    t = np.arange(n) / FS
    # Linear sweep. Phase is the integral of the instantaneous frequency.
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * secs))
    return (np.sin(phase) * gain).astype(np.float32)


def device_name(device):
    """A device's name, resolved once at startup so it can be re-found later.

    Indices are not stable: a Bluetooth sink that suspends comes back as a new
    PipeWire node, and the index PortAudio handed out at import time then
    addresses nothing (`Invalid number of channels`). The name survives.
    """
    if device is None:
        return None
    try:
        return sd.query_devices(device)['name']
    except Exception:
        return device if isinstance(device, str) else None


def _refresh():
    """Make PortAudio enumerate the devices again.

    Its device list is built at initialisation and never revisited, so a sink
    that appeared, vanished or moved since import is invisible to it. Nothing
    short of a restart of the library updates it.
    """
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        pass


def resolve(name):
    """Current index of the device with this name, or None for the default."""
    if name is None:
        return None
    for i, d in enumerate(sd.query_devices()):
        if d['name'] == name:
            return i
    for i, d in enumerate(sd.query_devices()):
        if name in d['name']:
            return i
    raise RuntimeError(f"dispositivo {name!r} nao esta mais na lista")


def wait_ready(name, timeout=20.0):
    """Block until the named sink advertises an output channel again.

    For a second or two after the previous process lets go of it, a Bluetooth
    sink is present and claims zero channels, and an open fails with `Invalid
    number of channels`. Retrying does not help; waiting does.
    """
    if name is None:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            i = resolve(name)
            if int(sd.query_devices(i)['max_output_channels']) > 0:
                return
        except Exception:
            pass
        time.sleep(1.0)
        _refresh()


def _fan_out(samples, device):
    """Mono up to however many channels the sink insists on.

    The node PortAudio finds when an idle Bluetooth sink wakes does not always
    advertise the channel count it had before, and asking for one channel can
    then fail with `Invalid number of channels`.
    """
    try:
        want = int(sd.query_devices(device)['max_output_channels'])
    except Exception:
        want = 1
    want = max(1, min(2, want))
    if want == 1:
        return samples
    return np.column_stack([samples] * want)


def play_and_record(samples, duration, in_name, out_name, attempts=3):
    """Start listening, then play, and keep the whole window.

    The recording starts first so nothing of the burst can be lost to the latency
    between the two streams, which on a Bluetooth sink is a substantial fraction
    of a second.

    Retried, because a suspended Bluetooth sink can fail the first open and
    succeed the second.
    """
    last = None
    for attempt in range(1, attempts + 1):
        try:
            wait_ready(out_name)
            in_device, out_device = resolve(in_name), resolve(out_name)
            out = _fan_out(samples, out_device)
            with Recorder(in_device) as rec:
                sd.play(out, FS, device=out_device)
                heard = rec.collect(duration)
                sd.stop()
            return heard
        except Exception as e:               # noqa: BLE001 - any device fault
            last = e
            print(f"[selfcapture] saida falhou ({type(e).__name__}: {e}); "
                  f"tentativa {attempt}/{attempts}", flush=True)
            time.sleep(2.0)
            # Re-enumerate before trying again: if the sink came back as a new
            # node, the old index will keep failing forever without this.
            _refresh()
    raise last


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=('fsk', 'mfsk', 'mary'), default='mary')
    ap.add_argument('--in-device', help="input device (index or name)")
    ap.add_argument('--out-device', help="output device (index or name)")
    ap.add_argument('--gain', type=float, default=0.5)
    ap.add_argument('--bytes', type=int, default=24)
    ap.add_argument('--text', help="send this exact text instead of random bytes")
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--trials', type=int, default=1)
    ap.add_argument('--fec', action='store_true')
    ap.add_argument('--repeat', type=int, default=2)
    ap.add_argument('--parallel', action='store_true')
    ap.add_argument('--grouped', action='store_true')
    ap.add_argument('--chord', action='store_true')
    ap.add_argument('--gap', type=float, default=None,
                    help="silencio no fim de cada simbolo mary, como fracao")
    ap.add_argument('--band', type=float, default=None,
                    help="anotado no JSON; nao afeta a transmissao")
    ap.add_argument('--chirp', nargs='?', const='300 6000 4', metavar='"f0 f1 s"',
                    help="grava uma varredura em vez de um payload")
    ap.add_argument('--sync-chirp', action='store_true',
                    help="prefixa a rajada com uma varredura curta, para o "
                         "receptor achar o inicio por filtro casado em vez de "
                         "convergir com o gate early/late")
    ap.add_argument('--sync-hush', type=float, default=0.03,
                    help="silencio entre a varredura e o primeiro simbolo")
    ap.add_argument('--tail', type=float, default=2.0,
                    help="segundos a gravar depois do fim da rajada")
    ap.add_argument('--link', default='auto',
                    help="como o alto-falante esta ligado: bluetooth, p2, "
                         "interno. Vai para o JSON -- um codec Bluetooth e uma "
                         "degradacao que o link real nao tem, e as duas coisas "
                         "nao podem ser media juntas")
    ap.add_argument('--out', default='captures')
    ap.add_argument('--label', default='')
    args = ap.parse_args()

    for name in ('in_device', 'out_device'):
        v = getattr(args, name)
        if v is not None and v.isdigit():
            setattr(args, name, int(v))
    # From here on the devices travel as names, not indices.
    args.in_device = device_name(args.in_device)
    args.out_device = device_name(args.out_device)
    print(f"[selfcapture] saida={args.out_device!r} entrada={args.in_device!r}",
          flush=True)

    common = dict(fs=FS, self_capture=True, link=args.link,
                  in_device=str(args.in_device), out_device=str(args.out_device),
                  gain=args.gain,
                  # Tabela Gray do 16-FSK (ver modem.gray_table); ausente = tabela antiga.
                  gray=GRAY_ATUAL)

    if args.chirp:
        f0, f1, secs = (float(x) for x in args.chirp.split())
        burst = build_chirp(f0, f1, secs, args.gain)
        with aviso.em_uso("microfone + alto-falante", "selfcapture: varredura"):
            heard = play_and_record(burst, secs + args.tail, args.in_device, args.out_device)
        stem = recording.save(args.out, heard, b'', kind='chirp',
                              label=args.label or 'self-chirp', mode=args.mode,
                              baud=0, chirp=[f0, f1, secs],
                              rms=float(np.sqrt(np.mean(np.square(heard)))),
                              peak=float(np.max(np.abs(heard))), **common)
        print(f"[selfcapture] {stem.name}  {len(heard)/FS:.1f}s")
        return

    baud = 1200 if args.mode == 'fsk' else MFSK_BAUD
    for trial in range(1, args.trials + 1):
        seed = args.seed if args.seed is not None else int(time.time() * 1000) & 0xFFFFFF
        payload = (args.text.encode() if args.text
                   else make_text_payload(seed, args.bytes))

        # A fresh modulator per trial: both carry phase across calls, and a
        # burst that began mid-phase is not the burst the live path sends.
        mod = build_modulator(args)
        burst = build_burst(args, mod, payload)
        airtime = len(burst) / FS
        with aviso.em_uso("microfone + alto-falante",
                          f"selfcapture: teste {trial}/{args.trials}, {airtime:.1f}s no ar"):
            heard = play_and_record(burst, airtime + args.tail,
                                    args.in_device, args.out_device)

        rms = float(np.sqrt(np.mean(np.square(heard)))) if len(heard) else 0.0
        peak = float(np.max(np.abs(heard))) if len(heard) else 0.0
        stem = recording.save(args.out, heard, payload,
                              label=args.label or f"self-{args.mode}",
                              mode=args.mode,
                              kind='fec' if args.fec else 'stream',
                              fec_repeat=args.repeat if args.fec else 0,
                              parallel=bool(args.fec and args.parallel),
                              gap=args.gap or 0.0, band=args.band or 0.0,
                              chord=bool(args.chord), grouped=bool(args.grouped),
                              sync_chirp=bool(args.sync_chirp),
                              sync_hush=args.sync_hush if args.sync_chirp else 0.0,
                              sync_span_symbols=args.sync_span_symbols,
                              baud=baud, seed=seed, rms=rms, peak=peak,
                              airtime_s=round(airtime, 2), **common)
        # Let the sink settle before the next trial: opening a Bluetooth output
        # stream right after closing one can fail while the device tears the previous
        # one down.
        time.sleep(1.0)
        print(f"[selfcapture] {trial}/{args.trials}  {stem.name}  "
              f"{len(heard)/FS:.1f}s  rms={rms:.4f} pico={peak:.3f}")
        if peak >= 0.99:
            print("             pico saturado -- baixe --gain")
        elif peak < 0.05:
            print("             quase nada chegou -- suba o volume ou aproxime o microfone")

    print(f"[selfcapture] pronto: {args.out}")


if __name__ == '__main__':
    main()
