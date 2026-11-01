"""On-disk format for captured audio, and the index over a directory of it.

A capture is two files sharing a stem: a `.wav` holding exactly what the
microphone delivered, and a `.json` holding what was transmitted and under
what conditions.

Samples are written as 32-bit float, the same dtype the demodulators consume,
so a capture goes back into the DSP bit-identical to what the live path saw.
`_wav_bytes` uses only `struct`, so recording never depends on a codec.

The archive format is 24-bit FLAC. Do not use 16-bit: a soft Viterbi near its
decoding limit needs only half a quantisation step to change a block's outcome.

`read` takes either format, and `audio_path` prefers the `.wav`, which is exact.
Writing FLAC needs `soundfile`; reading a `.wav` does not.

No audio device and no serial port: this module is disk only.
"""

import json
import struct
import time
from pathlib import Path

import numpy as np

FS = 48000


def _wav_bytes(samples, fs=FS):
    """Minimal 32-bit float mono WAV. `wave` in the stdlib cannot write these."""
    data = np.asarray(samples, dtype='<f4').tobytes()
    fmt = struct.pack('<HHIIHH', 3, 1, fs, fs * 4, 4, 32)   # IEEE float, mono
    chunks = (b'fmt ' + struct.pack('<I', len(fmt)) + fmt
              + b'data' + struct.pack('<I', len(data)) + data)
    return b'RIFF' + struct.pack('<I', 4 + len(chunks)) + b'WAVE' + chunks


def read_wav(path):
    raw = Path(path).read_bytes()
    i = raw.find(b'data')
    if i < 0:
        raise ValueError(f"{path}: no data chunk")
    size = struct.unpack('<I', raw[i + 4:i + 8])[0]
    return np.frombuffer(raw[i + 8:i + 8 + size], dtype='<f4').astype(np.float64)


# Former private name, kept so existing importers keep working.
_read_wav = read_wav


def read_flac(path):
    """A FLAC capture back as float64, in the same scale `read_wav` returns."""
    import soundfile                      # optional: only the archive needs it
    samples, _ = soundfile.read(str(path), dtype='float64', always_2d=False)
    return np.asarray(samples, dtype=np.float64)


def write_flac(path, samples, fs=FS):
    """Archive a capture as 24-bit FLAC. See the module docstring for why 24."""
    import soundfile
    soundfile.write(str(path), np.asarray(samples, dtype=np.float64), fs,
                    format='FLAC', subtype='PCM_24')
    return Path(path)


def audio_path(stem):
    """The audio file belonging to `stem`, whichever format it is stored in.

    Prefers the WAV, which holds the exact samples, and falls back to the FLAC.
    """
    stem = Path(stem)
    for suffix in ('.wav', '.flac'):
        candidate = stem.with_name(stem.name + suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{stem}: no .wav and no .flac")


def read(path):
    """One capture's audio, from a `.wav` or a `.flac`."""
    path = Path(path)
    return read_flac(path) if path.suffix == '.flac' else read_wav(path)


def save(directory, samples, payload, **meta):
    """Write one capture. Returns the stem both files share.

    The name carries the label and the timestamp.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    label = meta.get('label') or meta.get('mode') or 'capture'
    # A dot in the label is not a file extension, and `with_suffix` cannot tell
    # the difference, so the names are built by concatenation.
    stem = directory / f"{stamp}-{label}"
    return save_as(stem, samples, payload, recorded=stamp, **meta)


def save_as(stem, samples, payload, **meta):
    """Write one capture under a name the caller chose.

    `save` picks the name from the clock and the label; this one takes it, so a
    capture received from another machine can keep its original name.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_name(stem.name + '.wav').write_bytes(
        _wav_bytes(samples, meta.get('fs', FS)))
    info = dict(meta)
    info.setdefault('recorded', time.strftime("%Y%m%d-%H%M%S"))
    info.update(payload_hex=payload.hex(), payload_len=len(payload),
                samples=len(samples))
    stem.with_name(stem.name + '.json').write_text(json.dumps(info, indent=2) + "\n")
    return stem


def as_int16(samples):
    """Samples as 16-bit PCM bytes, for the serial cable.

    Half the size of the on-disk float format. The samples came from an ADC of
    16 bits or fewer, so the conversion loses nothing the microphone delivered.
    """
    x = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    return np.round(x * 32767.0).astype('<i2').tobytes()


def from_int16(blob):
    """The inverse. Exact for audio that was 16-bit to begin with."""
    return np.frombuffer(blob, dtype='<i2').astype(np.float64) / 32767.0


def load(json_path):
    """One capture back as (samples, payload, meta)."""
    json_path = Path(json_path)
    meta = json.loads(json_path.read_text())
    payload = bytes.fromhex(meta['payload_hex'])
    stem = json_path.with_name(json_path.name[:-len('.json')])
    return read(audio_path(stem)), payload, meta


def load_all(directory):
    """Every capture in a directory, oldest first."""
    return [load(p) for p in sorted(Path(directory).glob('*.json'))]
