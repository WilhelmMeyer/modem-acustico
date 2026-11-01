import numpy as np
import scipy.signal as signal

import fec

class FSKModulator:
    def __init__(self, fs=48000, baud=1200, f_space=2200, f_mark=1200):
        self.fs = fs
        self.baud = baud
        self.f_space = f_space # 0
        self.f_mark = f_mark   # 1
        self.samples_per_symbol = int(fs / baud)
        self.phase = 0.0

    def reset(self):
        """Drop carrier phase. Only between streams -- resetting mid-stream
        splices a discontinuity into the carrier, which clicks and spreads
        energy out of band."""
        self.phase = 0.0

    def modulate_byte(self, byte_val):
        # UART framing: 1 start bit (0), 8 data bits (LSB first), 1 stop bit (1)
        bits = [0] # Start bit
        for i in range(8):
            bits.append((byte_val >> i) & 1)
        bits.append(1) # Stop bit
        return self.modulate_bits(bits)

    def modulate_bits(self, bits):
        # Continuous phase FSK
        samples = []
        for b in bits:
            f = self.f_mark if b == 1 else self.f_space
            t = np.arange(self.samples_per_symbol) / self.fs
            # Phase accumulation to keep continuous phase
            omega = 2 * np.pi * f
            symbol_samples = np.sin(omega * t + self.phase)
            self.phase += omega * self.samples_per_symbol / self.fs
            self.phase %= 2 * np.pi
            samples.append(symbol_samples)
        return np.concatenate(samples) if samples else np.array([])
        
    def modulate(self, data: bytes):
        chunks = []
        for b in data:
            chunks.append(self.modulate_byte(b))
        if chunks:
            return np.concatenate(chunks)
        return np.array([])

class FSKDemodulator:
    def __init__(self, fs=48000, baud=1200, f_space=2200, f_mark=1200, squelch=0.001):
        self.fs = fs
        self.baud = baud
        self.f_space = f_space
        self.f_mark = f_mark
        self.samples_per_symbol = int(fs / baud)
        self.squelch = squelch
        
        # Design filters
        nyq = 0.5 * fs
        # Bandpass filter around our frequencies (e.g., 800 Hz to 2600 Hz)
        self.bpf_b, self.bpf_a = signal.butter(4, [800 / nyq, 2600 / nyq], btype='band')
        
        # Discriminator delay
        center_f = (f_space + f_mark) / 2.0
        # 90 degree phase shift at center frequency
        self.delay = int(fs / (4 * center_f)) 
        if self.delay == 0:
            self.delay = 1
            
        # Lowpass filter for baseband (cut off around baud rate)
        self.lpf_b, self.lpf_a = signal.butter(4, (baud * 1.5) / nyq, btype='low')
        
        # State for streaming
        self.prev_samples = np.zeros(self.delay)
        self.lpf_state = signal.lfilter_zi(self.lpf_b, self.lpf_a)
        self.bpf_state = signal.lfilter_zi(self.bpf_b, self.bpf_a)
        
        # UART state machine
        self.state = 'IDLE' # IDLE, START, DATA, STOP
        self.bit_idx = 0
        self.current_byte = 0
        self.sample_count = 0
        self.last_bb = 0

        # Level metrics from the most recent block, for link tuning.
        # Measured before squelch and before bit decision, so they reflect
        # what the sound card actually delivered.
        self.input_rms = 0.0   # RMS of the raw input
        self.input_peak = 0.0  # peak of the raw input (clipping check)
        self.level_rms = 0.0   # RMS after the bandpass, i.e. in-band energy

    def reset(self):
        self.prev_samples = np.zeros(self.delay)
        self.lpf_state = signal.lfilter_zi(self.lpf_b, self.lpf_a)
        self.bpf_state = signal.lfilter_zi(self.bpf_b, self.bpf_a)
        self.state = 'IDLE'
        self.last_bb = 0
        self.input_rms = 0.0
        self.input_peak = 0.0
        self.level_rms = 0.0

    def demodulate(self, samples):
        # 1. Bandpass filter
        filtered, self.bpf_state = signal.lfilter(self.bpf_b, self.bpf_a, samples, zi=self.bpf_state)

        # Level metrics for tuning. Cheap, and only over this block.
        if len(samples):
            self.input_rms = float(np.sqrt(np.mean(np.square(samples))))
            self.input_peak = float(np.max(np.abs(samples)))
            self.level_rms = float(np.sqrt(np.mean(np.square(filtered))))

        # 2. Delay and multiply discriminator
        # Pad with previous samples to do delay
        padded = np.concatenate((self.prev_samples, filtered))
        delayed = padded[:-self.delay]
        self.prev_samples = padded[-self.delay:]
        
        mult = filtered * delayed
        
        # 3. Lowpass filter
        baseband, self.lpf_state = signal.lfilter(self.lpf_b, self.lpf_a, mult, zi=self.lpf_state)
        
        # Squelch: if the signal is weak, force baseband to positive (Idle/Mark).
        energy = filtered ** 2
        # `mult` is already proportional to energy, so an absolute threshold on
        # baseband does it without another stateful filter.
        baseband = np.where(np.abs(baseband) < self.squelch, 1.0, baseband)
        
        # Mark and Space mapping
        # For f_mark=1200, f_space=2200, delay=7 (at 48kHz). 
        # phase shift = 2 * pi * f * delay / fs
        # phase_mark = 2 * pi * 1200 * 7 / 48000 = 1.099 rad -> cos(1.099) > 0 (Mark is positive)
        # phase_space = 2 * pi * 2200 * 7 / 48000 = 2.015 rad -> cos(2.015) < 0 (Space is negative)
        
        # 4. UART recovery
        output_bytes = []
        
        for bb in baseband:
            if self.state == 'IDLE':
                # Look for falling edge (1 to 0 transition)
                if self.last_bb > 0 and bb <= 0:
                    self.state = 'START'
                    # Wait half a symbol to sample at the center of the start bit
                    self.sample_count = self.samples_per_symbol // 2
            
            elif self.state == 'START':
                self.sample_count -= 1
                if self.sample_count <= 0:
                    # Check if it's still 0 (valid start bit)
                    if bb <= 0:
                        self.state = 'DATA'
                        self.bit_idx = 0
                        self.current_byte = 0
                        self.sample_count = self.samples_per_symbol
                    else:
                        # False start
                        self.state = 'IDLE'
                        
            elif self.state == 'DATA':
                self.sample_count -= 1
                if self.sample_count <= 0:
                    bit_val = 1 if bb > 0 else 0
                    self.current_byte |= (bit_val << self.bit_idx)
                    self.bit_idx += 1
                    self.sample_count = self.samples_per_symbol
                    
                    if self.bit_idx == 8:
                        self.state = 'STOP'
                        
            elif self.state == 'STOP':
                self.sample_count -= 1
                if self.sample_count <= 0:
                    bit_val = 1 if bb > 0 else 0
                    if bit_val == 1:
                        # Valid stop bit
                        output_bytes.append(self.current_byte)
                    # Return to idle
                    self.state = 'IDLE'
            
            self.last_bb = bb
            
        return bytes(output_bytes)


# --- Multi-tone FSK ------------------------------------------------------
#
# A second physical layer, for links where amplitude cannot be trusted. Each
# symbol is a chord of several tones, so scaling the whole signal scales both
# alternatives and gain drops out of the decision. Several tones per symbol add
# frequency diversity: a tone in a null of the channel costs accuracy rather
# than the symbol.
#
# No 2nd or 3rd harmonic of one symbol's tone lands on a tone of the other
# symbol: distortion manufactures harmonics, and one falling on the opposite
# chord would be evidence for the wrong symbol. Within one chord it is harmless.
#
# The tones come in pairs, and each pair is one vote. Both members of a pair sit
# 200 Hz apart, so the channel treats them alike and the comparison between them
# survives tilt, distance and volume. Five pairs vote; the majority is the bit,
# so one loud spurious tone can only ever carry one vote.
#
# Polarity alternates along the band -- on pairs 0, 2 and 4 the lower tone
# means 0, on pairs 1 and 3 it means 1 -- so both chords end up with nearly the
# same mean frequency and a tilted channel favours neither bit.
MFSK_PAIRS = (
    # (tone for bit 0, tone for bit 1)
    (700, 900),
    (1320, 1120),
    (1540, 1740),
    (2160, 1960),
    (2380, 2580),
)

MFSK_TONES_0 = tuple(p[0] for p in MFSK_PAIRS)
MFSK_TONES_1 = tuple(p[1] for p in MFSK_PAIRS)

# The other way to spend the same ten frequencies: all five low ones mean 0 and
# all five high ones mean 1, instead of interleaving pairs across the band. This
# gives up the equal mean frequency of the pairs for about a kilohertz of
# separation between the alternatives, where a pair has only 200 Hz. It needs a
# per-tone floor, or a tilted channel leans on one whole group.
def _schroeder(n):
    """Phase offsets that keep a chord's peak down without costing it power.

    Tones started at the same phase all crest together, so the sum peaks at N
    times one tone and must be divided by N to fit. Schroeder's quadratic
    offsets spread the crests.

    Only the transmitter cares: the receiver measures energy and is blind to
    phase, so this needs no coordination between the two ends. Off by default
    (`low_crest`).
    """
    return [np.pi * k * k / max(n, 1) for k in range(n)]


MFSK_LOW = tuple(sorted(t for p in MFSK_PAIRS for t in p)[:len(MFSK_PAIRS)])
MFSK_HIGH = tuple(sorted(t for p in MFSK_PAIRS for t in p)[len(MFSK_PAIRS):])

# A vote is a ratio, so five tones of pure noise still elect a bit. So the
# demodulator also has to know whether anything was transmitted at all. Each
# pair's losing tone is a frequency nobody sent: with a signal the winner towers
# over it, with noise alone the two are the same size.
MFSK_PRESENCE_MIN = 1.3   # winner/loser energy ratio, median across pairs


class MFSKModulator:
    """Multi-tone FSK. One chord per bit, UART 8N1 framing as above."""

    def __init__(self, fs=48000, baud=100, tones_0=MFSK_TONES_0,
                 tones_1=MFSK_TONES_1, parallel=False, grouped=False,
                 low_crest=False):
        if grouped:
            tones_0, tones_1 = MFSK_LOW, MFSK_HIGH
        # parallel=False: every pair sounds the same bit and the receiver
        # votes -- robust, one bit per symbol. parallel=True: each pair carries
        # its own bit, so a symbol carries as many bits as there are pairs.
        # Same symbol duration either way; the speed comes from spending the
        # frequencies on separate bits instead of all of them on one.
        self.parallel = parallel
        # Fixed per-tone phase offsets, so the tones of a chord do not all
        # crest at the same instant. Transmitter-only: the receiver measures
        # energy and cannot tell.
        self.low_crest = low_crest
        self._offsets = _schroeder(len(tones_0))
        self.fs = fs
        self.baud = baud
        self.tones_0 = tuple(tones_0)
        self.tones_1 = tuple(tones_1)
        self.samples_per_symbol = int(fs / baud)
        # Phase is carried per tone across calls, for the same reason the Bell
        # 202 modulator carries one: restarting a tone mid-stream splices a
        # discontinuity into it, which is a click and out-of-band energy.
        self.phase = {f: 0.0 for f in self.tones_0 + self.tones_1}

    def reset(self):
        for f in self.phase:
            self.phase[f] = 0.0

    def _symbol(self, bit):
        """One symbol: a single bit, or one bit per pair when parallel."""
        if self.parallel:
            tones = [(self.tones_1 if b else self.tones_0)[i]
                     for i, b in enumerate(bit)]
        else:
            tones = self.tones_1 if bit else self.tones_0
        n = np.arange(self.samples_per_symbol)
        out = np.zeros(self.samples_per_symbol)
        offs = self._offsets if self.low_crest else None
        for i, f in enumerate(tones):
            w = 2 * np.pi * f / self.fs
            out += np.sin(w * n + self.phase[f] + (offs[i] if offs else 0.0))
            self.phase[f] = (self.phase[f] + w * self.samples_per_symbol) % (2 * np.pi)
        # Normalised by chord size, so a chord never clips where a single tone
        # would not, and both chords carry equal power. With the crest offsets
        # the peak lands under that bound, so scale up to use the headroom.
        out = out / len(tones)
        peak = float(np.max(np.abs(out)))
        if self.low_crest and peak > 0:
            out = out * (0.98 / peak)
        return out

    def modulate_bits(self, bits):
        if self.parallel:
            k = len(self.tones_0)
            bits = list(bits)
            if len(bits) % k:                       # pad out the last symbol
                bits += [0] * (k - len(bits) % k)
            chunks = [self._symbol(bits[i:i + k]) for i in range(0, len(bits), k)]
        else:
            chunks = [self._symbol(b) for b in bits]
        return np.concatenate(chunks) if chunks else np.array([])

    def modulate_byte(self, byte_val):
        bits = [0]                                        # start
        bits += [(byte_val >> i) & 1 for i in range(8)]   # LSB first
        bits.append(1)                                    # stop
        return self.modulate_bits(bits)

    def modulate(self, data: bytes):
        chunks = [self.modulate_byte(b) for b in data]
        return np.concatenate(chunks) if chunks else np.array([])

    def idle(self, symbols):
        """Continuous mark. Timing recovery needs transitions, so a link-layer
        preamble should alternate rather than idle."""
        k = len(self.tones_0) if self.parallel else 1
        return self.modulate_bits([1] * symbols * k)


class MFSKDemodulator:
    """Energy-ratio detector with its own symbol clock.

    The Bell 202 demodulator above gets symbol timing for free: it hunts for
    the falling edge of a UART start bit in a continuous waveform. Energy over
    a window has no such edge to find, so this one recovers timing explicitly,
    with an early/late gate steered by the very contrast it uses to decide
    bits -- contrast peaks when the window sits inside one symbol and collapses
    when it straddles two.
    """

    def __init__(self, fs=48000, baud=100, tones_0=MFSK_TONES_0,
                 tones_1=MFSK_TONES_1, guard=0.15, contrast_min=0.3,
                 parallel=False, grouped=False, floor_alpha=0.02,
                 steer=True, skip=0, period=None):
        if grouped:
            tones_0, tones_1 = MFSK_LOW, MFSK_HIGH
        self.grouped = grouped
        # Frozen clock, as in `MaryDemodulator`: `steer=False` takes every
        # window one `period` after the last, starting at absolute sample
        # `skip`, with no early/late gate. Only correct once something else has
        # found the frame, such as the two sync sweeps.
        self.steer = steer
        self.skip = int(skip)
        self.period = None if period is None else float(period)
        self.floor_alpha = floor_alpha
        self.fs = fs
        self.baud = baud
        self.tones_0 = tuple(tones_0)
        self.tones_1 = tuple(tones_1)
        self.samples_per_symbol = int(fs / baud)
        self.contrast_min = contrast_min
        self.parallel = parallel

        # Skip the head of every symbol, where the previous symbol's
        # reverberation is still ringing.
        self.guard = int(guard * self.samples_per_symbol)

        n = np.arange(self.guard, self.samples_per_symbol)
        self.probe_0 = np.exp(-2j * np.pi * np.outer(self.tones_0, n) / fs)
        self.probe_1 = np.exp(-2j * np.pi * np.outer(self.tones_1, n) / fs)

        # No per-tone equaliser here: a running average of a tone's own energy
        # would absorb the previous symbol's decaying tail and treat it as that
        # tone's normal level. Pairing does the job without memory.

        # Early/late gate geometry. The on-time window starts `delta` into the
        # buffer, leaving margin on each side to look earlier and later.
        self.delta = max(1, self.samples_per_symbol // 8)
        self.step = max(1, self.samples_per_symbol // 32)

        self.reset()

    def reset(self):
        self.buf = np.zeros(0, dtype=np.float64)
        self.state = 'IDLE'
        self.bit_idx = 0
        self.current_byte = 0
        self.last_bit = 1
        self.framing_errors = 0
        self.input_rms = 0.0
        self.input_peak = 0.0
        self.contrast = 0.0
        self.consumed = 0
        self.last_window = 0
        self.frozen_k = 0
        # Only the grouped reading uses this. Comparing the sum of five low
        # tones against five high ones is an amplitude comparison, so it needs
        # each tone measured against what that frequency looks like when
        # nothing is sent there; pairing never did, because its two members sit
        # 200 Hz apart and the channel treats them alike.
        self.tone_floor = np.zeros(len(self.tones_0) + len(self.tones_1))

    def _score(self, start):
        """Bit and vote margin for the symbol window beginning at `start`.

        Five pairs of tones, one vote each: within a pair the two frequencies
        are 200 Hz apart, so the channel treats them alike and whichever
        arrives stronger names the bit. The majority wins. Do not sum the
        chords instead, which would let amplitude buy the answer.
        """
        seg = self.buf[start + self.guard:start + self.samples_per_symbol]
        e0 = np.abs(self.probe_0 @ seg) ** 2
        e1 = np.abs(self.probe_1 @ seg) ** 2
        if not np.any(e0) and not np.any(e1):
            # Exact digital silence: nothing to compare, so no contrast.
            return 1, 0.0, (np.zeros(len(self.tones_0)) if self.parallel
                            else 0.0)

        if self.grouped:
            return self._score_grouped(e0, e1)

        votes = e1 > e0
        ones = int(np.count_nonzero(votes))
        bit = 1 if ones * 2 > len(votes) else 0

        # The vote decides the bit; it must not also steer the clock. A tally
        # of five is quantised (the margin can only be 0.2, 0.6 or 1.0), and
        # the early/late gate needs to tell "almost centred" from "centred".
        # Steer on the per-pair contrast instead, which is continuous, and a
        # ratio so it means the same at any volume.
        contrast = float(np.mean(np.abs(e1 - e0) / np.maximum(e1 + e0, 1e-30)))

        # Did anyone actually transmit? Each pair's losing tone is a frequency
        # that was not sent, so the winner/loser ratio separates a real symbol
        # from a room decoding its own noise. The median ignores the pair that
        # happened to land in a null.
        win = np.where(votes, e1, e0)
        lose = np.where(votes, e0, e1)
        # The soft value the error-correcting decoder wants: how far each pair
        # leaned, summed. A pair that split 51/49 and one that split 99/1 are
        # the same vote and very different evidence. Each pair's term is a
        # ratio, so this stays amplitude-independent.
        per_pair = (e1 - e0) / np.maximum(e1 + e0, 1e-30)
        llr = per_pair if self.parallel else float(np.sum(per_pair))

        presence = float(np.median(win / np.maximum(lose, 1e-30)))
        if presence < MFSK_PRESENCE_MIN:
            return bit, 0.0, llr

        return bit, contrast, llr

    def _score_grouped(self, e0, e1):
        """Sum of the low group against the sum of the high group.

        Each tone is divided by its own running floor first, and that is
        required here: the two groups sit a kilohertz apart, so a tilt leans on
        one whole group, and in a sum nothing outvotes it.
        """
        e = np.concatenate([e0, e1])
        if not self.tone_floor.any():
            self.tone_floor[:] = e.mean()
        v = e / np.maximum(self.tone_floor, 1e-30)
        self.tone_floor += self.floor_alpha * (e - self.tone_floor)

        lo = float(v[:len(e0)].sum())
        hi = float(v[len(e0):].sum())
        total = lo + hi
        if total <= 0.0:
            return 1, 0.0, 0.0
        contrast = abs(hi - lo) / total
        llr = (hi - lo) / total
        bit = 1 if hi > lo else 0

        win, lose = (hi, lo) if bit else (lo, hi)
        if win / max(lose, 1e-30) < MFSK_PRESENCE_MIN:
            return bit, 0.0, llr
        return bit, contrast, llr

    def _feed_bit(self, bit, output):
        """UART 8N1 on the recovered bit stream, the same framing the Bell 202
        path uses, so either physical layer presents the same dumb serial line."""
        if self.state == 'IDLE':
            if bit == 0:
                self.state = 'DATA'
                self.bit_idx = 0
                self.current_byte = 0
        elif self.state == 'DATA':
            self.current_byte |= (bit << self.bit_idx)
            self.bit_idx += 1
            if self.bit_idx == 8:
                self.state = 'STOP'
        elif self.state == 'STOP':
            # Emit even when the stop bit is wrong, unlike the Bell 202 path.
            # Dropping it would turn one bad bit into a missing byte, and a
            # deletion shifts every byte after it. A substitution stays local,
            # and the CRC decides whether the packet is good.
            if bit != 1:
                self.framing_errors += 1
            output.append(self.current_byte)
            self.state = 'IDLE'

    def _symbols(self, samples):
        """Timing recovery and one decision per symbol, as a generator.

        Framed bytes and raw soft values are two readings of the same symbol
        stream, so they share this loop.
        """
        samples = np.asarray(samples, dtype=np.float64)
        if len(samples):
            self.input_rms = float(np.sqrt(np.mean(np.square(samples))))
            self.input_peak = float(np.max(np.abs(samples)))
        self.buf = np.concatenate((self.buf, samples))

        if not self.steer:
            period = self.period or float(self.samples_per_symbol)
            while True:
                # Absolute sample where window k belongs, rounded once, as in
                # `MaryDemodulator`: accumulating a rounded step would let the
                # rounding pile up into the very drift the period removes.
                want = int(round(self.skip + self.frozen_k * period))
                at = want - self.consumed
                if at < 0:
                    self.frozen_k += 1
                    continue
                if len(self.buf) < at + self.samples_per_symbol:
                    return
                bit, self.contrast, llr = self._score(at)
                self.last_bit = bit
                self.last_window = want
                self.frozen_k += 1
                self.buf = self.buf[at:]
                self.consumed += at
                yield bit, self.contrast, llr
            return

        need = self.samples_per_symbol + 2 * self.delta
        while len(self.buf) >= need:
            bit_e, c_e, l_e = self._score(0)
            bit_o, c_o, l_o = self._score(self.delta)
            bit_l, c_l, l_l = self._score(2 * self.delta)

            # Steer toward whichever gate sees the cleanest separation: best
            # contrast means the window sits most fully inside one symbol.
            if c_l > c_o and c_l >= c_e:
                adjust, bit, self.contrast, llr = self.step, bit_l, c_l, l_l
            elif c_e > c_o and c_e > c_l:
                adjust, bit, self.contrast, llr = -self.step, bit_e, c_e, l_e
            else:
                adjust, bit, self.contrast, llr = 0, bit_o, c_o, l_o

            self.last_bit = bit
            # Where this decision was actually measured, for the diagnostics.
            # The gate measures at an offset inside the buffer and then consumes
            # a different amount, so the buffer length cannot tell.
            self.last_window = self.consumed + (self.delta if adjust == 0
                                                else (0 if adjust < 0 else 2 * self.delta))
            step = self.samples_per_symbol + adjust
            self.consumed += step
            self.buf = self.buf[step:]
            yield bit, self.contrast, llr

    def demodulate(self, samples):
        output = []
        for bit, contrast, _llr in self._symbols(samples):
            # Amplitude-independent squelch. An absolute threshold would put
            # back the very dependence this layer exists to remove; contrast is
            # a ratio, so it means the same thing at any volume.
            if contrast < self.contrast_min:
                self.state = 'IDLE'
            else:
                self._feed_bit(bit, output)
        return bytes(output)

    def demodulate_soft(self, samples):
        """Log-likelihoods for the error-correcting decoder.

        One value per symbol when voting, one per pair per symbol when
        parallel. No squelch and no framing: a weak symbol is reported as a
        weak opinion, since dropping it would shorten the block and misalign
        everything after it.
        """
        vals = [llr for _b, _c, llr in self._symbols(samples)]
        if not vals:
            # Flat even when parallel, like every non-empty return, so that
            # callers can concatenate.
            return np.zeros(0)
        return np.array(vals).ravel() if self.parallel else np.array(vals)


# --- M-ary FSK: one tone at a time, four bits per tone ----------------------
#
# 16 tones, exactly one sounding at a time, and which tone it is names four
# bits. A chord of five tones has to be divided by five to stay inside the same
# peak amplitude, so each tone leaves the speaker 14 dB down; one tone at a time
# gets that back.
#
# A tone in a null of the room's comb response could never win a comparison.
# With 16 tones each is silent 15/16 of the time, so its long-run average energy
# is the noise floor at that frequency, and dividing by it before comparing makes
# a per-frequency calibrated detector.
MARY_TONES = (888, 1050, 1212, 1375, 1538, 1700, 1862, 2025,
              2188, 2350, 2512, 2675, 2838, 3000, 3162, 3325)
MARY_BITS = 4

# One nibble as a chord of three tones instead of a single one. Sixteen
# patterns, any two sharing at most one tone, so the distance between any two
# nibbles is four tone positions: mistaking one tone cannot change the nibble.
# Every tone is used in exactly three patterns.
#
# The trade is power against diversity: three tones must be divided by three to
# stay inside the same peak, costing 9.5 dB each. Off by default.
MARY_CODES = (
    (0, 2, 12), (0, 4, 6), (0, 8, 9), (1, 3, 14),
    (1, 6, 10), (1, 7, 15), (2, 6, 13), (2, 9, 14),
    (3, 5, 15), (3, 10, 11), (4, 5, 9), (4, 8, 15),
    (5, 12, 13), (7, 10, 13), (7, 11, 14), (8, 11, 12),
)


def _gray(i):
    return i ^ (i >> 1)


def _gray_inv(g):
    """The inverse of `_gray`: the i with _gray(i) == g."""
    n = g
    g >>= 1
    while g:
        n ^= g
        g >>= 1
    return n


# Neighbouring tones are the ones the channel confuses, so Gray coding makes
# that confusion cost one bit instead of up to four. What guarantees it is
# value = gray(tone): the tables below are indexed by value and give the tone,
# so the tone of value v is gray_inv(v). The older table, tone = gray(value),
# leaves three pairs of adjacent tones two bits apart.
#
# A wire change: transmitter and receiver must run the same table. Against
# each other the two tables swap the values of 12 of the 16 tones (only 0-3
# stay put), and the block fails its CRC like a bad channel. Recordings with no
# `gray` field in their JSON are read with the old table (`gray_gravado`).
GRAY_TOM = 'tom'        # value = gray(tone): adjacent tones one bit apart
GRAY_VALOR = 'valor'    # tone = gray(value): the older table
GRAY_ATUAL = GRAY_TOM   # what this code transmits; stamped into new recordings


def gray_table(bits, gray=GRAY_TOM):
    """Tone index for each value. `gray` names the mapping, see GRAY_TOM."""
    if gray == GRAY_TOM:
        return [_gray_inv(i) for i in range(1 << bits)]
    if gray == GRAY_VALOR:
        return [_gray(i) for i in range(1 << bits)]
    raise ValueError(f"gray={gray!r}: use {GRAY_TOM!r} ou {GRAY_VALOR!r}")


def gray_gravado(meta):
    """The Gray table a recording was made with, from its JSON sidecar.

    An absent field means the older table.
    """
    return meta.get('gray', GRAY_VALOR)


_GRAY = gray_table(MARY_BITS)
_UNGRAY = {g: i for i, g in enumerate(_GRAY)}
# The older table, for reading recordings made with it.
_GRAY_ANTIGO = gray_table(MARY_BITS, GRAY_VALOR)
_UNGRAY_ANTIGO = {g: i for i, g in enumerate(_GRAY_ANTIGO)}

# IFK by repetition tone: the tone sent is never the tone of the previous
# symbol. A seventeenth tone, one step above the last, means "same value as the
# previous symbol"; it is sent when the value repeats and the previous tone was
# not itself the repetition tone, and is never sent twice running. The receiver
# then leaves the tone it detected last out of the comparison, since the room's
# tail makes that the likeliest wrong answer. Only that tone is excluded, not
# its neighbours.
#
# Same 162.5 Hz step and the same probe geometry as the sixteen. Both ends must
# agree on it: a receiver without IFK cannot read the repetition tone.
MARY_REPEAT_TONE = 3487.5

# 2-FSK on the M-ary machinery: the same energy-per-tone detector divided by
# each tone's running floor, the same soft output, FEC, sync word, preamble,
# idle tail and optional sync sweeps, with two tones and one bit per symbol, at
# the same 100 baud.
#
# 1700 and 2350 Hz: 650 Hz apart is 5.5 bins of the 118 Hz window, and neither
# is the 2nd or 3rd harmonic of the other.
FSK2_TONES = (1700, 2350)
FSK2_BITS = 1

# Tone set and bits per symbol for each mode name that runs on the
# MaryModulator/MaryDemodulator pair. Every tool that rebuilds a receiver from
# a recording's `mode` reads it here, so a new layer cannot be scored with the
# wrong alphabet by a tool that forgot it.
TONE_LAYERS = {'mary': (MARY_TONES, MARY_BITS), 'fsk2': (FSK2_TONES, FSK2_BITS)}


def tone_layer(mode):
    """dict(tones=..., bits=...) for a mode in TONE_LAYERS."""
    tones, bits = TONE_LAYERS[mode]
    return dict(tones=tones, bits=bits)


def ifk_tones(values, gray=GRAY_TOM):
    """Tone index for each value under the IFK rule, starting a fresh frame.

    Index 16 is the repetition tone. One function for the transmitter and for
    every offline tool that has to know what was on the air; `gray` is the
    table the recording was made with (`gray_gravado`).
    """
    table = gray_table(MARY_BITS, gray)
    rep = 1 << MARY_BITS
    out, prev_v, prev_t = [], None, None
    for v in values:
        t = rep if (v == prev_v and prev_t != rep) else table[v]
        out.append(t)
        prev_v, prev_t = v, t
    return out


class MaryModulator:
    """One tone per symbol, at full amplitude. Four bits per symbol.

    With `gap` non-zero the tone stops early and the rest of the symbol is
    silence, a timing reference that does not depend on the data. Silence
    rather than a pilot tone, because a pilot's own tail would fill the gap it
    was meant to mark.
    """

    def __init__(self, fs=48000, baud=100, tones=MARY_TONES, gap=0.0,
                 chord=False, ifk=False, bits=MARY_BITS, gray=GRAY_TOM):
        if ifk and chord:
            raise ValueError("ifk e chord nao combinam: o IFK escolhe um tom")
        if bits != MARY_BITS and (ifk or chord):
            raise ValueError("ifk e chord so existem com 16 tons")
        if len(tones) != 1 << bits:
            raise ValueError(f"{len(tones)} tons para {bits} bits por simbolo")
        self.nbits = int(bits)
        # Tone of each value; `gray` only for re-synthesising an old burst.
        self.gray = gray_table(self.nbits, gray)
        self.fs = fs
        self.baud = baud
        self.ifk = bool(ifk)
        # The repetition tone is appended only when IFK is on, so with it off
        # the oscillator bank and every sample stay the same.
        self.tones = tuple(tones) + ((MARY_REPEAT_TONE,) if self.ifk else ())
        self.gap = gap
        self.chord = chord
        self.samples_per_symbol = int(fs / baud)
        # Samples the tone actually sounds for. The rest is the gap.
        self.samples_per_tone = self.samples_per_symbol - int(gap * self.samples_per_symbol)
        # Every tone's phase advances every symbol, whether or not it sounded,
        # so each behaves as a free-running oscillator switched on and off.
        # Restarting a tone's phase would splice in a click.
        self.phase = np.zeros(len(self.tones))
        self.restart_ifk()

    def reset(self):
        self.phase[:] = 0.0
        self.restart_ifk()

    def restart_ifk(self):
        """Forget the previous symbol, so the next one is sent absolute.

        Call at the head of every frame. The modulator outlives a burst, and a
        frame that opened with the repetition tone would be relative to a
        symbol the receiver never heard -- the idle tail of the last one.
        """
        self.prev_value = None
        self.prev_tone = None

    def _advance(self):
        w = 2 * np.pi * np.array(self.tones) / self.fs
        self.phase = (self.phase + w * self.samples_per_symbol) % (2 * np.pi)

    def _symbol(self, value):
        v = value & ((1 << self.nbits) - 1)
        nd = self.samples_per_tone
        n = np.arange(nd)
        if self.chord:
            idxs = MARY_CODES[v]
            out = np.zeros(nd)
            for i in idxs:
                w = 2 * np.pi * self.tones[i] / self.fs
                out += np.sin(w * n + self.phase[i])
            # Divided by the count, not its square root: the same peak budget
            # as a single tone, so the two schemes can be compared without one
            # of them quietly running hotter into the far side's limiter.
            out /= len(idxs)
        else:
            idx = self.gray[v]
            if (self.ifk and v == self.prev_value
                    and self.prev_tone != 1 << MARY_BITS):
                idx = 1 << MARY_BITS
            self.prev_value, self.prev_tone = v, idx
            w = 2 * np.pi * self.tones[idx] / self.fs
            out = np.sin(w * n + self.phase[idx])
        self._advance()
        if nd < self.samples_per_symbol:
            out = np.concatenate([out, np.zeros(self.samples_per_symbol - nd)])
        return out

    def modulate_bits(self, bits):
        bits = list(bits)
        nb = self.nbits
        if len(bits) % nb:
            bits += [0] * (nb - len(bits) % nb)
        chunks = []
        for i in range(0, len(bits), nb):
            v = 0
            for j, b in enumerate(bits[i:i + nb]):
                v |= (b & 1) << j
            chunks.append(self._symbol(v))
        return np.concatenate(chunks) if chunks else np.array([])

    def modulate(self, data: bytes):
        bits = []
        for byte in data:
            bits += [(byte >> i) & 1 for i in range(8)]
        return self.modulate_bits(bits)

    def idle(self, symbols):
        """A tail so the receiver's last symbols are not stranded in its
        buffer.

        Exactly `symbols` symbols at any bits per symbol: `fec.frame_symbols`
        counts the tail in symbols, and the two-sweep period divides by it.
        Below 16 tones it alternates symbol by symbol."""
        if self.nbits == MARY_BITS:
            return self.modulate_bits([0, 1, 0, 1] * symbols)
        return self.modulate_bits([(i // self.nbits) % 2
                                   for i in range(symbols * self.nbits)])


class MaryDemodulator:
    """Pick the loudest tone, after dividing each by its own noise floor."""

    def __init__(self, fs=48000, baud=100, tones=MARY_TONES, guard=0.15,
                 contrast_min=0.15, floor_alpha=0.02, gap=0.0, band=0.0,
                 chord=False, steer=True, skip=0, floor_fixed=None,
                 period=None, floor_norm=False, floor_clip=None, floor_top=1,
                 ifk=False, bits=MARY_BITS, gray=GRAY_TOM):
        if ifk and chord:
            raise ValueError("ifk e chord nao combinam: o IFK escolhe um tom")
        if bits != MARY_BITS and (ifk or chord):
            raise ValueError("ifk e chord so existem com 16 tons")
        if len(tones) != 1 << bits:
            raise ValueError(f"{len(tones)} tons para {bits} bits por simbolo")
        # Bits per symbol. Not `self.bits`, which is the leftover-bit list.
        self.nbits = int(bits)
        # `gray`: the table the transmitter used. GRAY_VALOR reads recordings
        # made with the older table (see `gray_gravado`).
        self.gray = gray_table(self.nbits, gray)
        self.ungray = {g: i for i, g in enumerate(self.gray)}
        # IFK by repetition tone: see `ifk_tones`. With it on, the tone this
        # demodulator detected in the previous symbol is left out of the
        # comparison, and a win by the repetition tone means "the value decided
        # last time". The running floor still updates every tone but the
        # loudest, the excluded one included.
        self.ifk = bool(ifk)
        self.rep = (1 << MARY_BITS) if self.ifk else None
        if self.ifk:
            tones = tuple(tones) + (MARY_REPEAT_TONE,)
        # `steer=False` freezes the symbol clock: every window is taken exactly
        # one symbol after the last, and `skip` says where the first one
        # starts. Only correct when something else has already found the burst.
        self.steer = steer
        self.skip = int(skip)
        # Samples per symbol as measured rather than nominal, and fractional on
        # purpose. Two sync bursts bracketing a frame give the interval the
        # symbols occupied, divided by their count. Rounding it would matter:
        # half a sample per symbol is a quarter of a symbol over 500 symbols.
        self.period = None if period is None else float(period)
        # A per-tone divisor supplied from outside instead of estimated. The
        # blind estimate assumes a tone's long-run average energy is its noise
        # floor, which holds because each tone is silent fifteen symbols in
        # sixteen.
        self.floor_fixed = (None if floor_fixed is None
                            else np.asarray(floor_fixed, dtype=np.float64))
        self.fs = fs
        self.baud = baud
        self.tones = np.array(tones, dtype=np.float64)
        self.samples_per_symbol = int(fs / baud)
        self.contrast_min = contrast_min
        self.floor_alpha = floor_alpha
        # Three knobs on the running floor:
        #
        #   floor_norm   divide each symbol's energies by their own mean
        #                before both the comparison and the floor update. A
        #                receive gain that moves during the burst then leaves
        #                the floor alone. Within a symbol the scale is a
        #                constant and cancels, so only the floor changes.
        #   floor_clip   ceiling, in multiples of the current floor, on what
        #                a single symbol may contribute to the update. The
        #                update leaves out the winner, not the transmitted
        #                tone, so a real tone that loses puts its full energy
        #                into its own floor, which makes it lose again.
        #   floor_top    how many of the loudest tones to leave out of each
        #                update. 1 is the winner alone.
        self.floor_norm = bool(floor_norm)
        self.floor_clip = None if floor_clip is None else float(floor_clip)
        # 0 means exclude nobody, only sensible together with `floor_clip`: the
        # ceiling keeps the winner's energy out of its own floor without the
        # selection bias of the exclusion (a tone that wins spuriously has its
        # noise peaks removed from its own floor, which makes it win again).
        self.floor_top = max(0, int(floor_top))
        self.gap = gap
        self.chord = chord

        # With a transmitted gap the tone occupies only the head of the symbol,
        # so measure there and leave the tail to the clock.
        self.samples_per_tone = (self.samples_per_symbol
                                 - int(gap * self.samples_per_symbol))
        self.guard = int(guard * self.samples_per_tone)

        # Measure a small band around each tone rather than the single exact
        # frequency: a misaligned window truncates the tone, which broadens it.
        # Keep `band` well under the 162 Hz to the neighbouring tone.
        n = np.arange(self.guard, self.samples_per_tone)
        self.band = band
        offsets = (0.0,) if not band else (-band, 0.0, band)
        self.probes = [np.exp(-2j * np.pi * np.outer(self.tones + o, n) / fs)
                       for o in offsets]
        self.probe = self.probes[0]

        self.delta = max(1, self.samples_per_symbol // 8)
        self.step = max(1, self.samples_per_symbol // 32)
        self.reset()

    def reset(self):
        self.buf = np.zeros(0, dtype=np.float64)
        self._e_key = None
        self._e_val = None
        self.pending_skip = 0
        self.frozen_k = 0
        self.floor = (np.zeros(len(self.tones)) if self.floor_fixed is None
                      else self.floor_fixed.copy())
        # Bits left over when a block does not end on a byte boundary. Audio
        # arrives in blocks holding a non-integer number of symbols, so a block
        # can end mid-byte, and dropping that half byte would shift every byte
        # after it by a nibble.
        self.bits = []
        self.input_rms = 0.0
        self.input_peak = 0.0
        self.contrast = 0.0
        # Absolute sample index of the head of the buffer, and of the window
        # the last decision was measured on. Diagnostics only. They cannot be
        # rebuilt from the buffer length, because the early/late gate measures
        # at an offset inside the buffer and then consumes a different amount.
        self.consumed = 0
        self.last_window = 0
        # IFK state: the tone detected and the value decided in the previous
        # symbol. `used_*` are what the symbol just yielded was decided
        # against, for the soft path, which runs after the state has moved on.
        self.prev_idx = None
        self.prev_val = 0
        self.used_excl = None
        self.used_ref = 0
        self.value = 0

    def _commit(self, idx):
        """Record the decision for the symbol about to be yielded.

        Called exactly once per symbol, just before the yield, and never from
        `_score` -- the steered path scores three candidate windows per symbol
        and all three must be judged against the same excluded tone.
        """
        self.used_excl, self.used_ref = self.prev_idx, self.prev_val
        if self.chord:
            v = idx
        elif self.ifk and idx == self.rep:
            v = self.prev_val
        else:
            v = self.ungray[idx]
        self.value = v
        self.prev_idx, self.prev_val = idx, v

    def _energies(self, start):
        # One-entry memo: this is most of the demodulator's cost and the same
        # window is asked for several times per symbol. The key names the
        # buffer's contents uniquely, since `consumed` only ever grows and
        # nothing changes `buf` without changing `consumed` or its length.
        key = (self.consumed, len(self.buf), start)
        if self._e_key == key:
            return self._e_val
        seg = self.buf[start + self.guard:start + self.samples_per_tone]
        if len(self.probes) == 1:
            e = np.abs(self.probe @ seg) ** 2
        else:
            e = sum(np.abs(p @ seg) ** 2 for p in self.probes)
        self._e_key, self._e_val = key, e
        return e

    def _gap_energy(self, start):
        """Power in the stretch the transmitter left silent.

        This is the timing reference, and its virtue is that it says nothing
        about the data -- it is low when the window is aligned and rises the
        moment the window slides onto a neighbouring tone. Steering on the
        decision's own contrast instead is circular, since a window that has
        lost the boundary also stops deciding well.
        """
        seg = self.buf[start + self.samples_per_tone:start + self.samples_per_symbol]
        return float(np.mean(np.square(seg))) if len(seg) else 0.0

    def _nibble_scores(self, norm):
        """How well each of the sixteen patterns explains this symbol.

        For single tones the pattern is the tone, so the score is the tone.
        For chords it is the sum over the three tones -- which is why a tone
        lost in a null costs a third of the evidence rather than all of it.
        """
        if not self.chord:
            return norm
        return np.array([norm[list(c)].sum() for c in MARY_CODES])

    def _scale(self, e):
        """Per-symbol divisor: 1.0 unless `floor_norm`, then the mean energy.

        Constant across the sixteen tones of one symbol, so it cancels out of
        every decision taken inside that symbol. What it does not cancel out
        of is the running floor, which is an average over many symbols at
        many receive gains.
        """
        if not self.floor_norm:
            return 1.0
        return max(float(np.mean(e)), 1e-30)

    def _score(self, start):
        e = self._energies(start)
        e = e / self._scale(e)
        # Divide by the running floor before comparing, so a tone in a null and
        # a tone on a peak are each judged against what they look like when
        # nobody is transmitting on them.
        norm = e / np.maximum(self.floor, 1e-30)
        scores = self._nibble_scores(norm)
        if self.ifk and self.prev_idx is not None:
            # The previous symbol's tone cannot be this symbol's, by
            # construction of the transmitter, and it is the tone the room's
            # tail is most likely to hand the decision. Out of the race.
            scores = scores.copy()
            scores[self.prev_idx] = -np.inf
        order = np.argsort(scores)
        top, second = scores[order[-1]], scores[order[-2]]
        contrast = (top - second) / max(top + second, 1e-30)
        return int(order[-1]), contrast, norm

    def _update_floor(self, e):
        if self.floor_fixed is not None:
            return
        # Each tone is silent 15 symbols out of 16, so its average energy is
        # its noise floor. The winner is left out of the update so the floor
        # never chases the signal it exists to measure.
        e = e / self._scale(e)
        mask = np.ones(len(self.tones), dtype=bool)
        if self.floor_top == 1:
            mask[int(np.argmax(e))] = False
        elif self.floor_top:
            mask[np.argsort(e)[-self.floor_top:]] = False
        if not self.floor.any():
            self.floor[:] = e.mean()
            return
        a = self.floor_alpha
        upd = e if self.floor_clip is None else np.minimum(
            e, self.floor_clip * self.floor)
        self.floor[mask] += a * (upd[mask] - self.floor[mask])

    def _symbols(self, samples):
        samples = np.asarray(samples, dtype=np.float64)
        if len(samples):
            self.input_rms = float(np.sqrt(np.mean(np.square(samples))))
            self.input_peak = float(np.max(np.abs(samples)))
        self.buf = np.concatenate((self.buf, samples))

        if self.pending_skip:
            drop = min(self.pending_skip, len(self.buf))
            self.buf = self.buf[drop:]
            self.consumed += drop
            self.pending_skip -= drop

        if not self.steer:
            period = self.period or float(self.samples_per_symbol)
            while True:
                # Absolute sample where window k belongs, rounded once at the
                # end. Accumulating a rounded step instead would let the error
                # pile up symbol after symbol, which is the drift this exists
                # to remove.
                want = int(round(self.skip + self.frozen_k * period))
                at = want - self.consumed
                if at < 0:                      # already past it; give up on k
                    self.frozen_k += 1
                    continue
                if len(self.buf) < at + self.samples_per_symbol:
                    return
                idx, contrast, norm = self._score(at)
                self.contrast = contrast
                self._update_floor(self._energies(at))
                self.last_window = want
                self.frozen_k += 1
                self.buf = self.buf[at:]
                self.consumed += at
                self._commit(idx)
                yield idx, contrast, norm
            return

        need = self.samples_per_symbol + 2 * self.delta
        while len(self.buf) >= need:
            i_e, c_e, n_e = self._score(0)
            i_o, c_o, n_o = self._score(self.delta)
            i_l, c_l, n_l = self._score(2 * self.delta)

            if self.gap:
                # Steer toward whichever window leaves the quietest gap. Lower
                # is better here, the opposite sense to contrast, so the scores
                # are negated and the same comparison below still applies.
                c_e = -self._gap_energy(0)
                c_o = -self._gap_energy(self.delta)
                c_l = -self._gap_energy(2 * self.delta)

            if c_l > c_o and c_l >= c_e:
                adjust, idx, norm, at = self.step, i_l, n_l, 2 * self.delta
                self.contrast = self._score(at)[1]
            elif c_e > c_o and c_e > c_l:
                adjust, idx, norm, at = -self.step, i_e, n_e, 0
                self.contrast = self._score(at)[1]
            else:
                adjust, idx, norm, at = 0, i_o, n_o, self.delta
                self.contrast = self._score(at)[1]

            self._update_floor(self._energies(at))
            self.last_window = self.consumed + at
            step = self.samples_per_symbol + adjust
            self.consumed += step
            self.buf = self.buf[step:]
            self._commit(idx)
            yield idx, self.contrast, norm

    def demodulate(self, samples):
        out = []
        for idx, contrast, _norm in self._symbols(samples):
            # In chord mode the winning index *is* the nibble, since the
            # patterns are indexed by value. With single tones it is a tone
            # index and Gray coding has to be undone first -- or, under IFK,
            # the repetition tone resolved to the previous value. `_commit`
            # did both.
            v = self.value
            self.bits += [(v >> j) & 1 for j in range(self.nbits)]
        while len(self.bits) >= 8:
            chunk = self.bits[:8]
            del self.bits[:8]
            out.append(sum(b << j for j, b in enumerate(chunk)))
        return bytes(out)

    def demodulate_soft(self, samples):
        """One log-likelihood per bit, max-log style.

        For each bit position, the best tone that would have carried a 1 is
        compared against the best that would have carried a 0. Working in the
        log domain means the comparison is between *ratios* of energy, so it
        means the same at any volume -- the property every decision in this
        project is built on.
        """
        out = []
        for _idx, _c, norm in self._symbols(samples):
            # Score the sixteen *nibbles*, not the sixteen tones. They differ
            # once a nibble is a chord: the evidence for a value is then the
            # sum over its three tones, and a bit's likelihood has to be read
            # off the values that carry it, whatever they are made of.
            log_e = np.log(np.maximum(self._nibble_scores(norm), 1e-30))
            nb = self.nbits
            gray = self.gray
            place = (lambda v: v) if self.chord else (lambda v: gray[v])
            if self.ifk:
                # Evidence for value v is the better of its own tone and, when
                # v is the value decided last symbol, the repetition tone --
                # the two ways the transmitter could have said it. The tone
                # detected last symbol is out, as in the hard decision, so the
                # hard decision is exactly the argmax of these sixteen.
                # A max rather than a log-sum: max-log like the rest of this
                # function, and the transmitter never sends both.
                log_e = log_e.copy()
                if self.used_excl is not None:
                    log_e[self.used_excl] = -np.inf
                metric = np.array([log_e[gray[v]] for v in range(1 << nb)])
                metric[self.used_ref] = max(metric[self.used_ref], log_e[self.rep])
                log_e = metric
                place = lambda v: v
            for j in range(nb):
                ones = [log_e[place(v)] for v in range(1 << nb) if (v >> j) & 1]
                zeros = [log_e[place(v)] for v in range(1 << nb) if not (v >> j) & 1]
                out.append(max(ones) - max(zeros))
        return np.array(out)


# --- Sync burst -------------------------------------------------------------
#
# An early/late gate acquires timing by converging over a preamble, and when it
# fails it costs a whole block that no coding rate can repair. A swept tone at
# the head of the frame correlates to one sharp peak instead, and that peak is
# an absolute sample index.
#
# Swept, not clicked: a click has the same detectability and a far worse crest
# factor, and an output limiter flattens it. A sweep spreads the same energy
# over its length, and its matched filter compresses it back to a point.

SYNC_CHIRP = (700.0, 3400.0, 0.08)     # f0, f1, seconds


def chirp(fs=48000, f0=SYNC_CHIRP[0], f1=SYNC_CHIRP[1], secs=SYNC_CHIRP[2]):
    """A linear sweep, as samples. Phase is the integral of the frequency.

    Keep the band inside the channel's usable one, 550-3500 Hz: energy where the
    channel returns none only dulls the correlation peak.
    """
    n = int(secs * fs)
    t = np.arange(n) / fs
    # A raised-cosine edge on each end. A sweep that starts and stops abruptly
    # is a sweep plus two clicks, and the clicks are what the limiter mangles.
    env = np.ones(n)
    edge = max(1, n // 16)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(edge) / edge))
    env[:edge] = ramp
    env[-edge:] = ramp[::-1]
    return np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * secs))) * env


def find_chirp(samples, template, threshold=4.0, leading_only=True):
    """Sample index just past the sweep, or None.

    Matched filter: correlate against the known sweep and take the peak. The
    score is the peak divided by the median of the correlation, so it is a
    ratio and means the same at any volume -- and the median, not the mean,
    because the peak itself would drag a mean upward and flatter the very
    thing being tested.

    Returns the index *after* the template, which is where the frame begins.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if len(samples) < len(template):
        return None
    corr = np.abs(np.correlate(samples, template, mode='valid'))
    med = float(np.median(corr))
    if med <= 0:
        return None
    peak = int(np.argmax(corr))
    if corr[peak] / med < threshold:
        return None
    if not leading_only:
        return peak + len(template)
    # A frame may carry a sweep at each end, and the far one is as likely to
    # be the louder. Take the earliest peak that clears the threshold, not the
    # tallest.
    first = int(np.argmax(corr / med >= threshold))
    # argmax on a boolean gives the first True; walk forward to the local peak
    # so the index lands on the sweep's centre rather than its leading edge.
    window = corr[first:first + len(template)]
    return first + int(np.argmax(window)) + len(template)


def find_chirp_pair(samples, template, min_gap, threshold=4.0):
    """Both sweeps bracketing a frame: (index after the first, index of the
    second), or None.

    One sweep gives the frame's start. Two give its start *and* its clock: the
    symbols between them occupied a known count and a measured interval, so
    the period follows by division instead of being tracked. Half a sample per
    symbol of error is a quarter of a symbol across a 500-symbol block, which
    is the whole difference between a block that decodes and one that does
    not, so the period is worth measuring rather than assuming.

    The peaks are found one at a time, the region around the first blanked
    before the second is sought -- a matched filter's response to a sweep is
    not a single sample wide, and taking the two largest values of the raw
    correlation returns the same peak twice.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if len(samples) < len(template) + min_gap:
        return None
    corr = np.abs(np.correlate(samples, template, mode='valid'))
    med = float(np.median(corr))
    if med <= 0:
        return None
    strongest = int(np.argmax(corr))
    if corr[strongest] / med < threshold:
        return None
    masked = corr.copy()
    lo = max(0, strongest - min_gap)
    hi = min(len(corr), strongest + min_gap)
    masked[lo:hi] = 0.0
    other = int(np.argmax(masked))
    if corr[other] / med < threshold:
        return None
    # Order by position, never by height. The two sweeps are identical and the
    # channel decides which arrives louder, and a reversed pair is a confident
    # wrong answer.
    first, second = sorted((strongest, other))
    return first + len(template), second


def sweep_soft(samples, make_demod, span, hush, sps, block=2048,
               template=None):
    """Soft values read with the clock the two sync sweeps measured.

    Returns (llr, skip, period, how). `how` is 'pair' when both sweeps were
    found and the period between them is plausible, 'lead' when only the
    leading sweep could be trusted (start known, nominal clock), and None
    when no sweep was found. Then llr, skip and period are None too and the
    caller falls back to the early/late gate.

    No I/O: `make_demod(steer=False, skip=..., period=...)` builds the receiver
    for the right layer, a tone layer's `MaryDemodulator` or a 5x2 layer's
    `MFSKDemodulator`, both of which take a frozen clock. `span` comes from
    `fec.sweep_span`, `hush` is the silence after each sweep in samples, `sps`
    the nominal samples/symbol.

    A period far from nominal means one of the peaks was not a sweep, so that
    case falls back to the leading sweep alone. Audio is fed in `block`-sized
    pieces because the demodulator carries state across calls.
    """
    samples = np.asarray(samples, dtype=np.float64)
    tmpl = chirp() if template is None else template
    skip = period = None
    how = None
    pair = find_chirp_pair(samples, tmpl, min_gap=int(0.5 * span * sps))
    if pair is not None:
        first, second = pair
        p = (second - first) / span
        if 0.98 * sps <= p <= 1.02 * sps:
            skip, period, how = first + int(round(hush * p / sps)), p, 'pair'
    if skip is None:
        at = find_chirp(samples, tmpl)
        if at is None:
            return None, None, None, None
        skip, how = at + hush, 'lead'
    d = make_demod(steer=False, skip=skip, period=period)
    parts = [d.demodulate_soft(samples[i:i + block])
             for i in range(0, len(samples), block)]
    parts = [q for q in parts if len(q)]
    llr = np.concatenate(parts) if parts else np.zeros(0)
    return llr, skip, period, how


def read_coded_frame(samples, make_demod, nbytes, repeat, layer, hush, sps,
                     sweeps, symbol_bits=None, npairs=None, gate_llr=None,
                     block=2048, template=None):
    """One coded frame off stored audio, the way every receiver here reads it.

    Returns (data, llr, how, period). `data` is None when the sync word was
    not found; `how` is 'pair', 'lead' or None as in `sweep_soft`, and None
    also means the early/late gate produced `llr`.

    The order is fixed: with `sweeps`, both sweeps (start and measured
    clock), then the leading sweep alone (start, nominal clock), then the
    gate; without, the gate only. Then the sync word and Viterbi through
    `fec.decode_block`, parallel when `layer` is 'mfsk-par'.

    `make_demod(**kw)` builds a fresh receiver for `layer`, with `kw` empty
    for the gate. `gate_llr`, when given, is the gate's reading already made
    by a streaming receiver, used instead of a second pass over `samples`.
    """
    samples = np.asarray(samples, dtype=np.float64)
    llr = how = period = None
    if sweeps:
        span = fec.sweep_span(nbytes, repeat, symbol_bits, hush, sps,
                              layer=layer, npairs=npairs)
        llr, _, period, how = sweep_soft(samples, make_demod, span, hush, sps,
                                         block=block, template=template)
    if llr is None:
        how = period = None
        if gate_llr is not None:
            llr = np.asarray(gate_llr, dtype=np.float64)
        else:
            d = make_demod()
            parts = [d.demodulate_soft(samples[i:i + block])
                     for i in range(0, len(samples), block)]
            parts = [q for q in parts if len(q)]
            llr = np.concatenate(parts) if parts else np.zeros(0)
    data, _ = fec.decode_block(llr, nbytes, repeat,
                               npairs=npairs if layer == 'mfsk-par' else None)
    return data, llr, how, period
