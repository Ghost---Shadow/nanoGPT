"""
Simulate simplified 1H-NMR time-domain signals (FIDs) for a handful of real
compounds, using approximate literature chemical shifts / J-couplings, then
turn each FID into a 2D spectrogram image (time x frequency) so it can be
fed through the same 2D block-DCT JPEG pipeline used for MNIST.

These are illustrative approximations (single reference solvent, simplified
first-order multiplets, no second-order effects) -- not lab-grade spectra.
"""
import numpy as np
from math import comb

SPEC_FREQ_MHZ = 300.0   # nominal spectrometer proton frequency
SAMPLE_RATE = 5000.0    # Hz
N_SAMPLES = 2048         # FID length -> ~0.41s acquisition, ~2.4Hz resolution

# compound -> list of (center_ppm, integration, multiplicity, J_Hz)
# multiplicity: 1=singlet, 2=doublet, 3=triplet, 4=quartet (first-order, binomial intensities)
COMPOUNDS = {
    'Water':      [(4.79, 2, 1, 0)],
    'Chloroform': [(7.26, 1, 1, 0)],
    'Benzene':    [(7.36, 6, 1, 0)],
    'Acetone':    [(2.17, 6, 1, 0)],
    'Methanol':   [(3.35, 3, 1, 0), (2.20, 1, 1, 0)],
    'Ethanol':    [(1.19, 3, 3, 7.0), (3.70, 2, 4, 7.0), (2.60, 1, 1, 0)],
}

def _expand_multiplet(center_ppm, integration, multiplicity, j_hz):
    center_hz = center_ppm * SPEC_FREQ_MHZ
    n = multiplicity
    coeffs = [comb(n - 1, i) for i in range(n)]
    total = sum(coeffs)
    lines = []
    for i, c in enumerate(coeffs):
        offset = (i - (n - 1) / 2) * j_hz
        lines.append((center_hz + offset, integration * c / total))
    return lines

def simulate_fid(compound, linewidth_hz=2.0, jitter=True, rng=None):
    rng = rng or np.random.default_rng()
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    fid = np.zeros(N_SAMPLES)
    for center_ppm, integration, multiplicity, j_hz in COMPOUNDS[compound]:
        shift_jitter = rng.normal(0, 0.02) if jitter else 0.0
        lw = linewidth_hz * (rng.uniform(0.8, 1.6) if jitter else 1.0)
        j = j_hz * (rng.uniform(0.95, 1.05) if jitter else 1.0)
        for freq_hz, amp in _expand_multiplet(center_ppm + shift_jitter, integration, multiplicity, j):
            phase = rng.uniform(0, 2 * np.pi) if jitter else 0.0
            amp_noise = amp * (rng.uniform(0.85, 1.15) if jitter else 1.0)
            t2 = 1.0 / (np.pi * lw)
            fid += amp_noise * np.cos(2 * np.pi * freq_hz * t + phase) * np.exp(-t / t2)
    if jitter:
        fid += rng.normal(0, 0.02 * (np.abs(fid).max() + 1e-6), size=N_SAMPLES)
    return fid

def _resize_2d(arr, out_h, out_w):
    h, w = arr.shape
    row_idx = (np.arange(out_h) * h / out_h).astype(int)
    col_idx = (np.arange(out_w) * w / out_w).astype(int)
    return arr[row_idx][:, col_idx]

def fid_to_image(fid, size=28, window=128, hop=32):
    n = len(fid)
    frames = []
    win = np.hanning(window)
    for start in range(0, n - window, hop):
        seg = fid[start:start + window] * win
        spec = np.abs(np.fft.rfft(seg))
        frames.append(spec)
    spectrogram = np.array(frames).T  # (freq_bins, time_frames)
    spectrogram = np.log1p(spectrogram)
    img = _resize_2d(spectrogram, size, size)
    img -= img.min()
    if img.max() > 0:
        img = img / img.max()
    return (img * 255).astype(np.uint8)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    for name in COMPOUNDS:
        fid = simulate_fid(name, rng=rng)
        img = fid_to_image(fid)
        print(f"{name:12s} fid_len={len(fid)} image shape={img.shape} range=[{img.min()},{img.max()}]")
