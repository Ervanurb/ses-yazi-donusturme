"""Ses ön işleme: yükleme, yeniden örnekleme, filtreleme, gürültü azaltma, VAD.

Tüm fonksiyonlar float32 numpy dizileri üzerinde çalışır (değerler -1..1 arası).
"""

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import av
import numpy as np
from scipy import signal
from scipy.io import wavfile

WHISPER_SR = 16_000  # Whisper 16 kHz mono ses bekler


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Herhangi bir ses/video dosyasını (wav, mp3, m4a, mp4...) okur.

    Örnekleme hızını DEĞİŞTİRMEZ; kanalları ortalamasını alarak mono yapar.
    Dönüş: (ses, örnekleme_hızı)
    """
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        sr = stream.rate
        # Sadece örnek biçimini float'a çeviriyoruz ("fltp" = planar float),
        # hız ve kanal düzeni aynı kalıyor.
        resampler = av.AudioResampler(format="fltp", layout=stream.layout.name, rate=sr)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())  # şekil: (kanal, örnek)
        for out in resampler.resample(None):  # tampondaki son örnekler
            chunks.append(out.to_ndarray())

    audio = np.concatenate(chunks, axis=1)
    mono = audio.mean(axis=0)  # stereo -> mono
    return mono.astype(np.float32), sr


def resample(audio: np.ndarray, sr_from: int, sr_to: int = WHISPER_SR) -> np.ndarray:
    """Polifaz filtreyle yeniden örnekleme (ör. 44100 -> 16000).

    44100 * 160/441 = 16000: önce 160 kat ara örnek eklenir, alçak geçiren
    filtreden geçirilir (aliasing'i önlemek için), sonra 441'de bir örnek alınır.
    """
    if sr_from == sr_to:
        return audio
    g = gcd(sr_from, sr_to)
    return signal.resample_poly(audio, sr_to // g, sr_from // g).astype(np.float32)


def normalize(audio: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    """En yüksek genliği peak_db dBFS'ye (varsayılan -1 dB) getirir."""
    peak = np.max(np.abs(audio))
    if peak < 1e-9:
        return audio
    target = 10 ** (peak_db / 20)
    return (audio * (target / peak)).astype(np.float32)


def highpass(audio: np.ndarray, sr: int, cutoff_hz: float = 80.0) -> np.ndarray:
    """Butterworth yüksek geçiren filtre: insan sesinin altındaki uğultuyu
    (elektrik hattı 50 Hz, rüzgar, masa titreşimi) keser.

    sosfiltfilt sinyali ileri ve geri iki kez filtreler, faz kayması olmaz.
    """
    sos = signal.butter(4, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def reduce_noise(
    audio: np.ndarray,
    sr: int,
    strength: float = 1.5,
    floor: float = 0.1,
    noise_percentile: float = 10.0,
) -> np.ndarray:
    """Spektral çıkarma ile durağan gürültüyü (fan, uğultu, hışırtı) azaltır.

    1. STFT ile sesi kısa zaman dilimlerinin frekans spektrumlarına ayır.
    2. En sessiz %10'luk dilimlerin ortalama spektrumunu "gürültü profili" say.
    3. Her frekans kutusunda, gürültü seviyesine yakın olan enerjiyi bastır:
          kazanç = 1 - strength * gürültü / genlik   (floor ile 1 arasında)
    4. Fazı koruyarak ters STFT ile sese geri dön.
    """
    nperseg = int(0.032 * sr)  # 32 ms pencere
    _, _, Z = signal.stft(audio, fs=sr, nperseg=nperseg)
    mag = np.abs(Z)

    frame_energy = mag.sum(axis=0)
    quiet = frame_energy <= np.percentile(frame_energy, noise_percentile)
    noise_profile = mag[:, quiet].mean(axis=1, keepdims=True)

    gain = 1.0 - strength * noise_profile / np.maximum(mag, 1e-10)
    gain = np.clip(gain, floor, 1.0)
    # Zaman ekseninde hafif yumuşatma: "müzikal gürültü" denen cıvıltıyı azaltır
    gain = signal.convolve2d(gain, np.ones((1, 3)) / 3, mode="same", boundary="symm")

    _, cleaned = signal.istft(Z * gain, fs=sr, nperseg=nperseg)
    return cleaned[: len(audio)].astype(np.float32)


@dataclass
class Segment:
    start: float  # saniye
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def frame_energy_db(audio: np.ndarray, sr: int, frame_ms: float = 30, hop_ms: float = 10):
    """Sesi örtüşen çerçevelere bölüp her birinin RMS enerjisini dB olarak verir."""
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if len(audio) < frame:
        audio = np.pad(audio, (0, frame - len(audio)))
    frames = np.lib.stride_tricks.sliding_window_view(audio, frame)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return 20 * np.log10(rms + 1e-10), hop / sr


def detect_speech(
    audio: np.ndarray,
    sr: int,
    margin_db: float = 12.0,
    min_silence: float = 0.3,
    min_speech: float = 0.2,
    pad: float = 0.1,
) -> list[Segment]:
    """Enerji tabanlı ses etkinliği tespiti (VAD).

    Eşik = arka plan gürültü seviyesi (en sessiz %10) + margin_db.
    Eşiği aşan çerçeveler konuşma sayılır; aradaki kısa sessizlikler
    birleştirilir, çok kısa parçalar (tık, nefes) atılır.
    """
    db, hop_s = frame_energy_db(audio, sr)
    noise_floor = np.percentile(db, 10)
    loud = np.percentile(db, 95)
    # Gürültü tabanı ile en yüksek seviye arasında makul bir eşik
    threshold = min(noise_floor + margin_db, loud - 6)
    active = db > threshold

    # Ardışık True bloklarını (başlangıç, bitiş) çiftlerine çevir
    edges = np.diff(np.concatenate([[0], active.astype(int), [0]]))
    starts = np.where(edges == 1)[0] * hop_s
    ends = np.where(edges == -1)[0] * hop_s

    segments: list[Segment] = []
    for s, e in zip(starts, ends):
        if segments and s - segments[-1].end < min_silence:
            segments[-1].end = e  # kısa sessizliği yut, öncekine ekle
        else:
            segments.append(Segment(s, e))

    total = len(audio) / sr
    return [
        Segment(max(0.0, seg.start - pad), min(total, seg.end + pad))
        for seg in segments
        if seg.duration >= min_speech
    ]


def preprocess(path: str | Path, denoise: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Dosyayı Whisper'a hazır hale getirir: 16 kHz, mono, filtreli, normalize.

    Dönüş: (orijinal_16k, işlenmiş_16k)
    """
    audio, sr = load_audio(path)
    audio = resample(audio, sr, WHISPER_SR)
    original = audio.copy()
    audio = highpass(audio, WHISPER_SR)
    if denoise:
        audio = reduce_noise(audio, WHISPER_SR)
    audio = normalize(audio)
    return original, audio


def save_wav(path: str | Path, audio: np.ndarray, sr: int = WHISPER_SR) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    wavfile.write(str(path), sr, pcm)
