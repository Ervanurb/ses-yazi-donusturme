"""Görselleştirme: dalga formu, konuşma bölgeleri ve log-mel spektrogram."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pencere açmadan dosyaya çiz
import matplotlib.pyplot as plt
import numpy as np

from .audio import WHISPER_SR, Segment

# Whisper'ın kullandığı değerler: 25 ms pencere, 10 ms adım, 80 mel bandı
N_FFT = 400
HOP = 160
N_MELS = 80


def hz_to_mel(f):
    return 2595 * np.log10(1 + np.asarray(f) / 700)


def mel_to_hz(m):
    return 700 * (10 ** (np.asarray(m) / 2595) - 1)


def mel_filterbank(sr: int = WHISPER_SR, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    """Üçgen mel filtreleri. Kulak düşük frekanslarda daha hassas olduğu için
    filtreler alt frekanslarda dar ve sık, üst frekanslarda geniş ve seyrektir."""
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fb = np.zeros((n_mels, len(fft_freqs)))
    for i in range(n_mels):
        left, center, right = hz_points[i : i + 3]
        rising = (fft_freqs - left) / (center - left)
        falling = (right - fft_freqs) / (right - center)
        fb[i] = np.maximum(0, np.minimum(rising, falling))
    return fb


def log_mel_spectrogram(audio: np.ndarray, sr: int = WHISPER_SR) -> np.ndarray:
    """Whisper'ın girdi olarak gördüğü log-mel spektrogramın aynısı (80 x zaman)."""
    window = np.hanning(N_FFT + 1)[:-1]
    padded = np.pad(audio, N_FFT // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP] * window
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2  # (zaman, frekans)
    mel = mel_filterbank(sr) @ power.T  # (80, zaman)
    log_spec = np.log10(np.maximum(mel, 1e-10))
    # Whisper gibi: en yüksek değerin 8 birim (80 dB) altını kırp
    return np.maximum(log_spec, log_spec.max() - 8.0)


def plot_analysis(
    original: np.ndarray,
    processed: np.ndarray,
    segments: list[Segment],
    out_path: str | Path,
    sr: int = WHISPER_SR,
    title: str = "",
) -> None:
    t = np.arange(len(original)) / sr
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(t, original, lw=0.4, color="tab:gray")
    axes[0].set_title("Orijinal ses (16 kHz mono)")

    axes[1].plot(t, processed, lw=0.4, color="tab:blue")
    for seg in segments:
        axes[1].axvspan(seg.start, seg.end, color="tab:green", alpha=0.2)
    axes[1].set_title(f"İşlenmiş ses — yeşil: tespit edilen konuşma ({len(segments)} bölüm)")

    mel = log_mel_spectrogram(processed, sr)
    duration = len(processed) / sr
    axes[2].imshow(mel, origin="lower", aspect="auto", cmap="magma", extent=[0, duration, 0, N_MELS])
    axes[2].set_title("Log-mel spektrogram (Whisper'ın 'gördüğü' girdi)")
    axes[2].set_ylabel("Mel bandı")
    axes[2].set_xlabel("Zaman (s)")

    for ax in axes[:2]:
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Genlik")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
