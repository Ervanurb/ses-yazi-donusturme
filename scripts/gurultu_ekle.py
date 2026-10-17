"""Temiz bir kayda gerçekçi bozulmalar ekleyerek test dosyası üretir:
50 Hz şebeke uğultusu + beyaz gürültü + başta/sonda sessizlik, 44.1 kHz stereo.

Kullanım: python scripts/gurultu_ekle.py samples/tts.wav samples/ornek_gurultulu.wav
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ses_yazi.audio import load_audio, normalize, resample, save_wav  # noqa: E402

SR = 44_100

src, dst = sys.argv[1], sys.argv[2]
audio, sr = load_audio(src)
audio = normalize(resample(audio, sr, SR), -6)

silence = np.zeros(int(1.5 * SR), dtype=np.float32)
audio = np.concatenate([silence, audio, silence])

rng = np.random.default_rng(0)
t = np.arange(len(audio)) / SR
hum = 0.05 * np.sin(2 * np.pi * 50 * t) + 0.02 * np.sin(2 * np.pi * 100 * t)
noise = 0.03 * rng.standard_normal(len(audio))
noisy = audio + hum + noise

# Sol ve sağ kanal biraz farklı olsun ki mono'ya indirme de test edilsin
stereo = np.stack([noisy, noisy * 0.9]).T.astype(np.float32)
save_wav(dst, stereo, SR)
print(f"Kaydedildi: {dst}")
