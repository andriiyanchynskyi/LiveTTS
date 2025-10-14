# enhancements.py
# Safe post-processing for XTTS waveform with optional flags.
# Input: wav float32/float64 in [-1, 1], sr (int), flags: Dict[str,bool]
# Output: wav float32 in [-1, 1]

from typing import Dict, Optional
import numpy as np

try:
    from scipy.signal import butter, sosfilt
    _SCIPY = True
except Exception:
    _SCIPY = False

import logging
log = logging.getLogger(__name__)

# Default parameters (tuned for speech)
HP_FREQ = 80.0       # Hz high-pass default
LP_FREQ = 7500.0     # Hz low-pass default
DEN_NOISE_FLOOR_DB = -60.0  # spectral floor for denoise
DEN_REDUCE_DB = 8.0         # reduction amount
NORM_TARGET_DBFS = -1.0     # normalization target peak

# -------------------------------------------------
# Optional audio-post-processing "enhancements"
# -------------------------------------------------
#
# Each key is a short flag the UI will send
#
ENHANCEMENTS: Dict[str, str] = {
    "sox": "Use SOX for speed control (pitch-preserving alternative to native XTTS, best for >1.5x or <0.7x)",
    "denoise":   "Noise reduction (spectral gating)",
    "normalize": "Loudness normalization (–1 dBFSS)",
    "equalize":  "Gentle EQ for fuller low-mids & crisp highs",
    "highpass":  "High-pass filter @ 80 Hz (remove rumble)",
    "lowpass":   "Low-pass filter @ 7.5 kHz (tame hiss)",
}

def _early_exit(flags: Optional[Dict[str, bool]]) -> bool:
    if not flags:
        return True
    return not any(flags.values())

def _as_float_mono(wav: np.ndarray) -> np.ndarray:
    # Ensure float32 mono in [-1, 1]
    if wav.ndim > 1:
        wav = wav[:, 0]
    if wav.dtype != np.float32 and wav.dtype != np.float64:
        wav = wav.astype(np.float32)
    # clamp if needed (XTTS gives already normalized float)
    wav = np.clip(wav, -1.0, 1.0)
    return wav.astype(np.float32)

def _peak_normalize(wav: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    peak = np.max(np.abs(wav)) if wav.size else 0.0
    if peak <= 1e-9:
        return wav
    # target linear from dBFS
    target_lin = 10.0 ** (target_dbfs / 20.0)
    gain = target_lin / peak
    out = wav * gain
    # avoid clipping
    out = np.clip(out, -1.0, 1.0)
    return out

def _butter_sos(kind: str, cutoff: float, sr: int, order: int = 2):
    # kind: 'highpass' or 'lowpass'
    nyq = 0.5 * sr
    w = max(min(cutoff / nyq, 0.999), 1e-5)
    if kind == 'highpass':
        sos = butter(order, w, btype='highpass', output='sos')
    else:
        sos = butter(order, w, btype='lowpass', output='sos')
    return sos

def _iir_filter(wav: np.ndarray, sr: int, kind: str, cutoff: float, order: int = 2) -> np.ndarray:
    if not _SCIPY:
        # no scipy: return input unmodified
        log.warning("scipy not installed; %s filter bypassed", kind)
        return wav
    if cutoff <= 0.0 or cutoff >= sr * 0.49:
        return wav
    sos = _butter_sos(kind, cutoff, sr, order=order)
    return sosfilt(sos, wav)

def _log_spectral_denoise(wav: np.ndarray, sr: int,
                          noise_floor_db: float = DEN_NOISE_FLOOR_DB,
                          reduce_db: float = DEN_REDUCE_DB) -> np.ndarray:
    """
    Lightweight denoise: log-magnitude gate per short-window.
    It avoids speech over-suppression used in stronger NR libs on float signals.
    """
    # STFT params
    win = 1024
    hop = 256
    if wav.size < win:
        return wav

    # Hann window
    window = np.hanning(win).astype(np.float32)

    # Pad to frame-aligned
    pad = (win - wav.size % hop) % hop
    x = np.pad(wav, (0, pad), mode='constant')

    # Frames
    frames = []
    for i in range(0, len(x) - win + 1, hop):
        frames.append(x[i:i+win] * window)
    frames = np.stack(frames, axis=1)  # [win, frames]

    # FFT
    spec = np.fft.rfft(frames, axis=0)
    mag = np.abs(spec)
    phase = np.angle(spec)

    # Estimate noise floor from lower-percentile per-bin
    # Use 15th percentile across time as background
    prc = np.percentile(mag, 15, axis=1, keepdims=True) + 1e-8

    # Build suppression mask in linear domain from dB thresholds
    floor_lin = 10.0 ** (noise_floor_db / 20.0)
    reduce_lin = 10.0 ** (-abs(reduce_db) / 20.0)

    # Normalize by noise floor, clamp to [reduce_lin, 1.0]
    norm = mag / np.maximum(prc, floor_lin)
    mask = np.clip(norm, reduce_lin, 1.0)

    # Apply mask
    mag_d = mag * mask

    # iFFT
    spec_d = mag_d * np.exp(1j * phase)
    frames_out = np.fft.irfft(spec_d, axis=0).real

    # Overlap-add
    out = np.zeros(len(x), dtype=np.float32)
    wsum = np.zeros(len(x), dtype=np.float32)
    for idx, i in enumerate(range(0, len(x) - win + 1, hop)):
        out[i:i+win] += frames_out[:, idx] * window
        wsum[i:i+win] += window**2

    nz = wsum > 1e-6
    out[nz] /= wsum[nz]
    out = out[:len(wav)]

    # Safety clamp
    out = np.clip(out, -1.0, 1.0)
    return out.astype(np.float32)

def _presence_tone_shaper(wav: np.ndarray, sr: int) -> np.ndarray:
    """
    Mild low-cut rumble + gentle presence lift ~3 kHz.
    Implemented with two IIR filters (HP + mild shelf emulation via LP complement).
    """
    # 1) Mild rumble removal
    y = _iir_filter(wav, sr, 'highpass', HP_FREQ, order=2)

    # 2) Presence region (approx 2–5 kHz) gentle lift
    # Emulate shelf: y + alpha * (y - lowpass(y, 2500 Hz))
    # This boosts content above cutoff slightly.
    cutoff = 2500.0
    lp = _iir_filter(y, sr, 'lowpass', cutoff, order=2)
    alpha = 0.10  # ~ +1 dB subjective
    y2 = y + alpha * (y - lp)

    # Clamp
    return np.clip(y2, -1.0, 1.0).astype(np.float32)

def apply_enhancements(wav: np.ndarray,
                       sr: int,
                       flags: Optional[Dict[str, bool]] = None) -> np.ndarray:
    """
    Master entry. Applies only explicitly enabled enhancements.
    Each step is light-weight and tuned for speech.
    Order: highpass/lowpass (if chosen) -> denoise -> eq -> normalize
    """
    if _early_exit(flags):
        return _as_float_mono(wav)

    y = _as_float_mono(wav)

    # 1) Optional linear filters (very low order to avoid phase damage)
    if flags.get("highpass", False):
        y = _iir_filter(y, sr, 'highpass', HP_FREQ, order=2)

    if flags.get("lowpass", False):
        y = _iir_filter(y, sr, 'lowpass', LP_FREQ, order=2)

    # 2) Optional gentle denoise
    if flags.get("denoise", False):
        try:
            y = _log_spectral_denoise(y, sr,
                                      noise_floor_db=DEN_NOISE_FLOOR_DB,
                                      reduce_db=DEN_REDUCE_DB)
        except Exception as e:
            log.warning("Denoise failed, bypassed: %s", e)

    # 3) Optional presence EQ (safe, speech-oriented)
    if flags.get("equalize", False):
        try:
            y = _presence_tone_shaper(y, sr)
        except Exception as e:
            log.warning("Equalize failed, bypassed: %s", e)

    # 4) Optional peak normalization near -1 dBFS
    if flags.get("normalize", False):
        y = _peak_normalize(y, target_dbfs=NORM_TARGET_DBFS)

    # Safety
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    return y
