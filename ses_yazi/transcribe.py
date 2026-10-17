"""Whisper (faster-whisper) ile konuşmayı yazıya dökme ve çıktı biçimleri."""

import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .audio import WHISPER_SR, Segment


def _add_cuda_dlls() -> None:
    """pip ile kurulan nvidia-cublas / nvidia-cudnn DLL'lerini Windows'ta bulunur yapar.
    (Sistemde CUDA Toolkit kurmaya gerek kalmasın diye.)"""
    if sys.platform != "win32":
        return
    import site

    for base in site.getsitepackages():
        nvidia = Path(base) / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in nvidia.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


def load_model(size: str | None = None, device: str = "auto"):
    """Modeli yükler. GPU varsa float16 ile GPU'da, yoksa int8 ile CPU'da çalışır.

    Model boyutları: tiny, base, small, medium, large-v3, turbo (large-v3-turbo).
    İlk çalıştırmada model Hugging Face'ten indirilir ve önbelleğe alınır.
    """
    _add_cuda_dlls()
    # Modeller herkese açık; kayıtlı (belki süresi dolmuş) HF token'ını gönderme
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    import ctranslate2
    from faster_whisper import WhisperModel

    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

    if device == "cuda":
        try:
            model = WhisperModel(size or "turbo", device="cuda", compute_type="float16")
            # cuDNN gibi kütüphaneler ilk kullanımda yüklenir; hatayı şimdi yakala
            list(model.transcribe(np.zeros(WHISPER_SR, dtype=np.float32), language="tr")[0])
            return model, "cuda"
        except Exception as e:  # noqa: BLE001
            print(f"[uyarı] GPU kullanılamadı, CPU'ya geçiliyor: {e}", file=sys.stderr)

    return WhisperModel(size or "small", device="cpu", compute_type="int8"), "cpu"


@dataclass
class Line:
    start: float
    end: float
    text: str


@dataclass
class Result:
    lines: list[Line]
    language: str
    language_prob: float
    seconds: float  # işlem süresi

    @property
    def text(self) -> str:
        return " ".join(line.text for line in self.lines)


def transcribe(
    model,
    audio: np.ndarray,
    speech: list[Segment] | None = None,
    language: str | None = None,
    beam_size: int = 5,
) -> Result:
    """16 kHz mono sesi yazıya döker.

    speech verilirse yalnızca bu konuşma bölümleri işlenir (clip_timestamps):
    Whisper sessiz kısımları hiç görmez, hem hızlanır hem de sessizlikte
    "uydurma" metin (halüsinasyon) üretme riski azalır. Zaman damgaları
    yine orijinal kayda göre gelir.
    """
    clips = None
    if speech:
        clips = [t for seg in speech for t in (seg.start, seg.end)]

    t0 = time.perf_counter()
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        clip_timestamps=clips if clips else "0",
        condition_on_previous_text=False,  # bir hatanın zincirleme yayılmasını önler
    )
    lines = [Line(s.start, s.end, s.text.strip()) for s in segments]  # generator burada çalışır
    return Result(lines, info.language, info.language_probability, time.perf_counter() - t0)


def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(result: Result) -> str:
    blocks = [
        f"{i}\n{_srt_time(line.start)} --> {_srt_time(line.end)}\n{line.text}\n"
        for i, line in enumerate(result.lines, 1)
    ]
    return "\n".join(blocks)


def to_txt(result: Result, timestamps: bool = False) -> str:
    if not timestamps:
        return "\n".join(line.text for line in result.lines) + "\n"
    return "\n".join(f"[{line.start:7.2f} - {line.end:7.2f}] {line.text}" for line in result.lines) + "\n"


def to_dict(result: Result) -> dict:
    return {
        "language": result.language,
        "language_probability": round(result.language_prob, 3),
        "processing_seconds": round(result.seconds, 2),
        "lines": [asdict(line) for line in result.lines],
    }
