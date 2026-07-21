"""
Turn an MNIST digit image into a short sequence of integers using a
JPEG-style pipeline: block DCT-II, quantization, zigzag scan, trailing-zero
truncation (like JPEG's End-Of-Block marker) per block.

This is the actual JPEG transform-coding step (frequency-domain quantized
DCT coefficients) -- not the final Huffman-coded byte stream, and not
wavelets (that would be JPEG2000). It's what the user was describing:
each image becomes a short sequence of small integers.

block_shape defaults to the standard 8x8 (giving a 4x4 = 16 block grid on
the 32x32-padded image) using the official JPEG luminance quant table. Any
other block_shape (e.g. (4, 8) for a 8x4 = 32 block grid) uses a synthetic
frequency-radial quant table generalized to arbitrary block dimensions,
since the official table is only defined for 8x8.
"""
import numpy as np

# standard JPEG luminance quantization table (quality ~50), 8x8 only
Q50_8x8 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)

def _quality_scale(quality):
    quality = max(1, min(100, quality))
    return 5000 / quality if quality < 50 else 200 - quality * 2

def quality_table(quality, block_shape=(8, 8)):
    scale = _quality_scale(quality)
    bh, bw = block_shape
    if block_shape == (8, 8):
        base = Q50_8x8
    else:
        # synthetic radial-frequency table: quantization step grows with
        # distance from the DC (top-left) corner, same spirit as the
        # official table (fine near DC, coarse at high frequency)
        u = np.arange(bh).reshape(-1, 1) / bh
        v = np.arange(bw).reshape(1, -1) / bw
        base = 10 + 100 * np.sqrt(u ** 2 + v ** 2)
    table = np.floor((base * scale + 50) / 100)
    return np.clip(table, 1, 255)

_dct_mat_cache = {}
def _dct_matrix(n):
    if n not in _dct_mat_cache:
        u = np.arange(n).reshape(-1, 1)
        x = np.arange(n).reshape(1, -1)
        C = np.where(u == 0, np.sqrt(1 / n), np.sqrt(2 / n))
        _dct_mat_cache[n] = C * np.cos((2 * x + 1) * u * np.pi / (2 * n))
    return _dct_mat_cache[n]

_zigzag_cache = {}
def _zigzag_order(bh, bw):
    if (bh, bw) not in _zigzag_cache:
        _zigzag_cache[(bh, bw)] = sorted(
            ((r, c) for r in range(bh) for c in range(bw)),
            key=lambda rc: (rc[0] + rc[1], -rc[1] if (rc[0] + rc[1]) % 2 else rc[1]))
    return _zigzag_cache[(bh, bw)]

def dct2d(block):
    bh, bw = block.shape
    return _dct_matrix(bh) @ block @ _dct_matrix(bw).T

def image_to_int_sequence(img, quality=50, block_shape=(8, 8), return_blocks=False):
    """img: 28x28 uint8 array. Returns a flat list of ints (one JPEG-style
    quantized-DCT sequence, blocks concatenated left-to-right, top-to-bottom,
    each block truncated after its last non-zero coefficient).
    If return_blocks, also returns a list of (block_row, block_col, start, end)
    giving each spatial block's [start, end) index range within the flat list."""
    bh, bw = block_shape
    assert 32 % bh == 0 and 32 % bw == 0, "block_shape must evenly divide the 32x32 padded image"
    img = np.asarray(img, dtype=np.float64)
    padded = np.zeros((32, 32))
    padded[:28, :28] = img
    padded -= 128.0  # standard JPEG level shift
    qtable = quality_table(quality, block_shape)
    zigzag = _zigzag_order(bh, bw)

    seq = []
    block_ranges = []
    for br, by in enumerate(range(0, 32, bh)):
        for bc, bx in enumerate(range(0, 32, bw)):
            block = padded[by:by + bh, bx:bx + bw]
            coeffs = np.round(dct2d(block) / qtable).astype(int)
            zz = [int(coeffs[r, c]) for r, c in zigzag]
            # trim trailing zeros (JPEG's End-Of-Block), keep at least the DC term
            last_nonzero = 0
            for i, v in enumerate(zz):
                if v != 0:
                    last_nonzero = i
            start = len(seq)
            seq.extend(zz[:last_nonzero + 1])
            block_ranges.append((br, bc, start, len(seq)))
    if return_blocks:
        return seq, block_ranges
    return seq


if __name__ == '__main__':
    # quick sanity check on a synthetic image (no MNIST download needed for this)
    fake = np.zeros((28, 28), dtype=np.uint8)
    fake[8:20, 12:16] = 255  # a vertical bar, vaguely "1"-shaped
    for shape in [(8, 8), (4, 8)]:
        seq = image_to_int_sequence(fake, block_shape=shape)
        n_blocks = (32 // shape[0]) * (32 // shape[1])
        print(f"block_shape={shape} -> {n_blocks} blocks, sequence length {len(seq)}")
