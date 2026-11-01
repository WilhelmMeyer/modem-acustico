"""Forward error correction: convolutional coding with soft-decision Viterbi.

The decoder takes the demodulator's analog confidence rather than hard bits, so
a doubtful decision weighs less than a confident one. Inside a coded block there
is no UART framing: under 8N1 one corrupted start or stop bit shifts every byte
after it, and a fixed-length block has nothing to shift.

No audio, no I/O, no state: bits and log-likelihoods in, bytes out.
"""

import numpy as np

# Constraint length 7, the standard NASA/Voyager polynomials.
K = 7
POLYS_R12 = (0o171, 0o133)
POLYS_R13 = (0o171, 0o133, 0o165)

NSTATES = 1 << (K - 1)


def _parity(x):
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 1


def _tables(polys):
    """Output bits and next state for every (state, input) pair.

    Built in one place so the encoder and the decoder walk the same trellis.
    """
    out = np.zeros((NSTATES, 2, len(polys)), dtype=np.int8)
    nxt = np.zeros((NSTATES, 2), dtype=np.int32)
    for state in range(NSTATES):
        for bit in (0, 1):
            reg = (bit << (K - 1)) | state
            for i, p in enumerate(polys):
                out[state, bit, i] = _parity(reg & p)
            nxt[state, bit] = reg >> 1
    return out, nxt


_TABLES = {POLYS_R12: _tables(POLYS_R12), POLYS_R13: _tables(POLYS_R13)}


def encode_bits(bits, polys=POLYS_R12):
    """Convolutionally encode, flushing the register at the end.

    The tail of K-1 zero bits returns the encoder to state 0, so the decoder
    knows where it finished.
    """
    out_tab, nxt_tab = _TABLES[polys]
    state = 0
    coded = []
    for bit in list(bits) + [0] * (K - 1):
        coded.extend(out_tab[state, bit])
        state = nxt_tab[state, bit]
    return np.array(coded, dtype=np.int8)


def decode_soft(llr, nbits, polys=POLYS_R12):
    """Viterbi over the trellis, taking log-likelihood ratios.

    `llr` is positive where the channel leans towards a 1 and negative
    towards 0, with magnitude meaning confidence. A hard decision is the
    special case where every magnitude is the same.

    Vectorised over the 64 states so the decoder keeps up with the audio.
    """
    out_tab, nxt_tab = _TABLES[polys]
    n = len(polys)
    steps = nbits + K - 1
    llr = np.asarray(llr, dtype=np.float64)[:steps * n].reshape(-1, n)
    if len(llr) < steps:
        raise ValueError(f"precisa de {steps * n} llrs, recebi {llr.size}")

    # Branch metric: agreement between the expected output bits and the
    # channel's opinion. Expected bit 1 wants positive llr, bit 0 negative.
    signs = 2 * out_tab.astype(np.float64) - 1          # (state, input, n)

    INF = 1e18
    metric = np.full(NSTATES, -INF)
    metric[0] = 0.0                                      # encoder starts at 0
    back = np.zeros((steps, NSTATES), dtype=np.int8)

    prev_for = np.zeros((NSTATES, 2), dtype=np.int32)
    inp_for = np.zeros((NSTATES, 2), dtype=np.int8)
    seen = np.zeros(NSTATES, dtype=np.int8)
    for state in range(NSTATES):
        for bit in (0, 1):
            ns = nxt_tab[state, bit]
            prev_for[ns, seen[ns]] = state
            inp_for[ns, seen[ns]] = bit
            seen[ns] += 1

    for t in range(steps):
        # Score of each (state, input) branch against this step's llrs.
        branch = (signs * llr[t]).sum(axis=2)             # (state, input)
        cand = np.empty((NSTATES, 2))
        for j in (0, 1):
            src = prev_for[:, j]
            cand[:, j] = metric[src] + branch[src, inp_for[:, j]]
        best = np.argmax(cand, axis=1)
        metric = cand[np.arange(NSTATES), best]
        back[t] = best
        metric -= metric.max()                            # keep it bounded

    # The tail drove the encoder back to state 0, so that is where to start.
    bits = np.zeros(steps, dtype=np.int8)
    state = 0
    for t in range(steps - 1, -1, -1):
        j = back[t, state]
        prev = prev_for[state, j]
        bits[t] = inp_for[state, j]
        state = prev
    return bits[:nbits]


def pair_map(ncoded, repeat, npairs):
    """Where each coded bit goes, when every tone pair carries its own bit.

    Copy r of coded bit i goes on pair (i + r) mod npairs. The coded length is
    a multiple of the pair count, so plain tiling would put every copy of a bit
    on the same pair, and one pair in a null would take them all down. Copies
    laid down pass by pass also land far apart in time.

    Returns one entry per transmitted slot, holding the coded-bit index that
    slot carries, or -1 for padding.
    """
    cols = [[] for _ in range(npairs)]
    for r in range(repeat):
        for i in range(ncoded):
            cols[(i + r) % npairs].append(i)
    nsym = max(len(c) for c in cols)
    order = np.full(nsym * npairs, -1, dtype=np.int64)
    for p, col in enumerate(cols):
        order[p:len(col) * npairs:npairs] = col
    return order


def frame_parallel(data, npairs, polys=POLYS_R13, repeat=1):
    """Sync word then the coded block, laid out across the pairs."""
    coded = encode_bits(bytes_to_bits(data), polys)
    order = pair_map(len(coded), repeat, npairs)
    body = np.where(order >= 0, coded[np.maximum(order, 0)], 0).astype(np.int8)
    # The sync word goes out on every pair at once, the way the preamble does:
    # it has to be findable before anything is known about which pairs work.
    sync = np.repeat(SYNC, npairs)
    return np.concatenate([sync, body])


def decode_parallel(llr, nbytes, npairs, polys=POLYS_R13, repeat=1):
    """Gather the copies back onto their coded bits, then decode.

    Adding the copies is optimal combining -- independent observations of one
    bit add in the log-likelihood domain -- so a copy that landed on a good
    pair outweighs one that landed in a null without any thresholding.
    """
    nbits = nbytes * 8
    ncoded = (nbits + K - 1) * len(polys)
    order = pair_map(ncoded, repeat, npairs)

    llr = np.asarray(llr, dtype=np.float64)[:len(order)]
    if len(llr) < len(order):
        llr = np.concatenate([llr, np.zeros(len(order) - len(llr))])

    # No per-pair normalisation here: the repetition map keeps any one pair from
    # carrying more than one copy of a bit, so a pair stuck in a null is
    # outnumbered by construction. `find_sync_parallel` does need it.
    acc = np.zeros(ncoded)
    keep = order >= 0
    np.add.at(acc, order[keep], llr[keep])
    return bits_to_bytes(decode_soft(acc, nbits, polys))


def find_sync_parallel(llr, npairs, threshold=0.35):
    """Block start when the sync word was sent on every pair at once.

    The sync word is the one part of a parallel block where all the pairs
    carry the same bit, so it is read by voting: average the pairs, then
    correlate one value per symbol. Matching position by position would let a
    pair in a null, which answers the same whatever was sent, hold the score
    under the threshold.

    Each pair is centred first. Over a scrambled block a pair's soft values
    average to zero, so any remaining offset is the channel, not information.
    """
    llr = np.asarray(llr, dtype=np.float64)
    nsym = len(llr) // npairs
    if nsym < len(SYNC):
        return None
    m = llr[:nsym * npairs].reshape(nsym, npairs)
    m = m - m.mean(axis=0)
    per_symbol = m.mean(axis=1)

    want = 2.0 * SYNC - 1.0
    scores = np.correlate(np.sign(per_symbol), want, mode='valid') / len(SYNC)
    best = int(np.argmax(scores))
    if scores[best] < threshold:
        return None
    return (best + len(SYNC)) * npairs


def interleave_index(n, depth):
    """Block interleaver as a permutation, so both directions share one map.

    Spreads consecutive coded bits apart, turning a run of errors into isolated
    ones, which is what a convolutional code handles well.
    """
    rows = int(np.ceil(n / depth))
    idx = np.arange(rows * depth).reshape(rows, depth).T.ravel()
    return idx[idx < n]


# 31-bit reference word, found by correlating against the soft stream. Its
# linear autocorrelation has peak 31 and sidelobes within +-7. The block start
# cannot be found by counting symbols, because the early/late gate consumes a
# varying number of samples per symbol as it steers.
SYNC = np.array([1,0,0,1,0,1,0,1,1,0,0,1,1,1,1,1,
                 0,0,0,1,1,0,1,1,1,0,1,0,1,0,0], dtype=np.int8)


def frame(data, polys=POLYS_R13, depth=16, repeat=1):
    """Sync word followed by the coded block, as bits ready to modulate."""
    return np.concatenate([SYNC, encode(data, polys, depth, repeat)])


def preamble_bits(mode, symbol_bits=4, npairs=1, parallel=False, symbols=120):
    """The alternating run that precedes a coded block, as bits.

    Timing recovery is an early/late gate steered by decision contrast, so it
    locks onto transitions and learns nothing from identical symbols. Hence
    alternating, and long enough to converge before the sync word arrives.

    Bits in, bits out: which tones carry them is the modulator's business.
    """
    if mode == 'mary':
        # Alternate between the two extreme tones: value 0 against all ones.
        out = []
        for i in range(symbols):
            v = 0 if i % 2 else (1 << symbol_bits) - 1
            out += [(v >> j) & 1 for j in range(symbol_bits)]
        return out
    # MFSK sends the same bit on every pair at once, so the far side can lock
    # before it knows which pairs are working.
    return [0, 1] * 40 * (npairs if parallel else 1)


def find_sync(llr, threshold=0.55):
    """Where the block starts in a stream of soft values, or None.

    Correlates against the sync word and takes the sharpest peak. The
    threshold is on the normalised score, so it means the same at any signal
    level.
    """
    llr = np.asarray(llr, dtype=np.float64)
    if len(llr) < len(SYNC):
        return None
    want = 2.0 * SYNC - 1.0
    scores = np.correlate(np.sign(llr), want, mode='valid') / len(SYNC)
    best = int(np.argmax(scores))
    if scores[best] < threshold:
        return None
    return best + len(SYNC)


def bytes_to_bits(data):
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder='little')


def bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    bits = bits[:len(bits) // 8 * 8]
    return np.packbits(bits, bitorder='little').tobytes()


def encode(data, polys=POLYS_R13, depth=16, repeat=1):
    """Bytes in, coded bits out: encoded, repeated, then interleaved.

    `repeat` extends the code past what the polynomials alone reach, since
    copies of a bit combine by addition in the log-likelihood domain.

    Interleaving comes after repetition so that copies of a bit end up far
    apart in time, and a burst that destroys one leaves the others intact.
    """
    coded = encode_bits(bytes_to_bits(data), polys)
    if repeat > 1:
        coded = np.tile(coded, repeat)
    return coded[interleave_index(len(coded), depth)]


def decode(llr, nbytes, polys=POLYS_R13, depth=16, repeat=1):
    """Coded llrs in, bytes out. `nbytes` is the length the sender used."""
    nbits = nbytes * 8
    ncoded = (nbits + K - 1) * len(polys)
    total = ncoded * repeat

    llr = np.asarray(llr, dtype=np.float64)[:total]
    if len(llr) < total:
        llr = np.concatenate([llr, np.zeros(total - len(llr))])

    deint = np.zeros(total)
    deint[interleave_index(total, depth)] = llr
    # Summing the copies is optimal combining: for independent observations of
    # one bit the log-likelihoods add.
    return bits_to_bytes(decode_soft(deint.reshape(repeat, ncoded).sum(axis=0),
                                     nbits, polys))


def frame_symbols(nbytes, repeat, symbol_bits, idle_symbols=6):
    """Symbols a coded frame occupies: preamble, body, trailing idle.

    Transmitter and receiver must agree on this count: the interval between the
    two sync sweeps is divided by it to get the symbol period, and a span one
    symbol off accumulates to a whole symbol of drift over a long block.

    The body rounds up, because the modulator pads the coded bits out to a
    whole symbol.
    """
    pre = preamble_bits('mary', symbol_bits=symbol_bits)
    nbits = len(frame(bytes(nbytes), repeat=repeat))
    return len(pre) // symbol_bits + -(-nbits // symbol_bits) + idle_symbols


# The two 5x2 layers. Their coded frames are built differently from the tone
# layers' and so occupy a different number of symbols: see `chord_frame_symbols`.
CHORD_LAYERS = ('mfsk', 'mfsk-par')


def chord_frame_symbols(nbytes, repeat, npairs, parallel, idle_symbols=4):
    """Symbols a coded frame occupies on the 5x2 layers (voted or parallel).

    Not `frame_symbols`: these frames open with the 80-symbol alternation of
    `preamble_bits('mfsk')` rather than 120, close on `idle(4)` rather than 6,
    and the parallel body is `frame_parallel` laid five bits to a symbol. The
    wrong count can still fall inside the receiver's 2% period window, where it
    is accepted silently and drifts.

    Preamble and body each round up on their own, because the modulator pads
    each `modulate_bits` call out to a whole symbol separately.
    """
    pre = preamble_bits('mfsk', npairs=npairs, parallel=parallel)
    if parallel:
        nbits = len(frame_parallel(bytes(nbytes), npairs, repeat=repeat))
        return -(-len(pre) // npairs) + -(-nbits // npairs) + idle_symbols
    return len(pre) + len(frame(bytes(nbytes), repeat=repeat)) + idle_symbols


def layer_frame_symbols(layer, nbytes, repeat, symbol_bits=None, npairs=None):
    """Symbols a coded frame occupies on `layer`, whichever layer it is.

    `symbol_bits` is what the tone layers ('mary', 'fsk2') need, `npairs`
    what the 5x2 layers ('mfsk', 'mfsk-par') need. Transmitter and receivers
    must share this count.
    """
    if layer in CHORD_LAYERS:
        if not npairs:
            raise ValueError(f"camada {layer} precisa de npairs")
        return chord_frame_symbols(nbytes, repeat, npairs, layer == 'mfsk-par')
    if not symbol_bits:
        raise ValueError(f"camada {layer} precisa de symbol_bits")
    return frame_symbols(nbytes, repeat, symbol_bits)


def sweep_span(nbytes, repeat, symbol_bits, hush, sps, layer=None, npairs=None):
    """Symbols between the two sync-sweep detections, as transmitted.

    The coded frame plus the two silences that separate it from the sweeps,
    in symbols so it divides straight into the measured interval. `hush` is in
    samples, `sps` in samples per symbol. `layer` is needed only for the 5x2
    layers; left out, the frame is a tone layer's.
    """
    if layer in CHORD_LAYERS:
        body = layer_frame_symbols(layer, nbytes, repeat, npairs=npairs)
    else:
        body = frame_symbols(nbytes, repeat, symbol_bits)
    return body + 2 * hush / sps


def decode_block(llr, nbytes, repeat, npairs=None):
    """Sync word, then Viterbi: (bytes, start), or (None, None) without sync.

    `npairs` set means a parallel 5x2 block, whose sync word is read by voting
    across the pairs (`find_sync_parallel`) and whose copies are gathered off
    their pairs (`decode_parallel`); unset, every other coded frame.
    """
    llr = np.asarray(llr, dtype=np.float64)
    if npairs:
        start = find_sync_parallel(llr, npairs)
        if start is None:
            return None, None
        return decode_parallel(llr[start:], nbytes, npairs, repeat=repeat), start
    start = find_sync(llr)
    if start is None:
        return None, None
    return decode(llr[start:], nbytes, repeat=repeat), start
