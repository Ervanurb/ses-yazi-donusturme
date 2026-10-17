# ses-yazi

Sesi yazıya dönüştüren küçük bir araç. Konuşma tanıma için [faster-whisper](https://github.com/SYSTRAN/faster-whisper) kullanıyor. Whisper'dan önceki ses işleme adımlarını ise öğrenmek amacıyla elle yazdık.

```
ses dosyası ─► yükle (PyAV) ─► mono ─► 16 kHz'e yeniden örnekle ─► yüksek geçiren filtre (80 Hz)
          ─► spektral gürültü azaltma ─► normalize ─► enerji tabanlı VAD ─► Whisper ─► txt / srt / json
```

## Kurulum

Python 3.10–3.12 gerekir.

```bash
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

NVIDIA ekran kartı varsa model otomatik olarak GPU'da çalışır. CUDA Toolkit kurmaya gerek yok, gerekli DLL'ler pip paketlerinden yükleniyor. GPU yoksa CPU'ya geçilir. Modeller ilk çalıştırmada indirilir (turbo yaklaşık 1,6 GB, small yaklaşık 500 MB).

## Kullanım

**Web arayüzü** (dosya yükleme veya mikrofonla kayıt):

```bash
.venv\Scripts\python app.py
```
Ardından tarayıcıda http://127.0.0.1:7860 adresini açın.

**Komut satırı:**

```bash
.venv\Scripts\python transcribe.py kayit.mp3                 # txt + srt + json -> output/
.venv\Scripts\python transcribe.py a.mp3 b.m4a --dil tr --grafik
.venv\Scripts\python transcribe.py kayit.mp3 --model small --cihaz cpu
```

**Yalnızca analiz** (Whisper olmadan, ön işlemeyi görmek için):

```bash
.venv\Scripts\python analyze.py kayit.mp3                    # temiz wav + grafik -> output/
```

## Dosyalar

| Dosya | İçerik |
|---|---|
| `ses_yazi/audio.py` | Yükleme, yeniden örnekleme, filtre, gürültü azaltma, VAD |
| `ses_yazi/viz.py` | Mel filtre bankası, log-mel spektrogram (Whisper ile aynı), grafikler |
| `ses_yazi/transcribe.py` | Model yükleme (GPU/CPU), yazıya dökme, SRT/TXT/JSON |
| `transcribe.py`, `analyze.py`, `app.py` | Komut satırı ve web arayüzü |
| `scripts/` | Windows'un Türkçe sesiyle test kaydı üretme, kayda gürültü ekleme |

## Not

`samples/` içindeki kayıtlar Windows'un yapay "Tolga" sesiyle üretildi. Whisper bu robotik sesi zor tanıyor, gerçek insan sesinde doğruluk çok daha yüksek.
