# ses-yazi dönüştürücü

Ses kayıtlarını metne dönüştüren, Türkçe odaklı bir konuşma tanıma aracı. Kayıt önce temizleniyor, sonra konuşma olan bölümler bulunuyor, en sonunda bu bölümler [Whisper](https://github.com/openai/whisper) modeliyle yazıya dökülüyor. Komut satırından ya da tarayıcıdaki bir arayüzden kullanılabiliyor. Çıktı olarak düz metin, altyazı (`.srt`) ve JSON üretiyor.

## Projenin amacı

Bu proje, **ses işlemeyi öğrenmek** için geliştirildi.

Konuşma tanıma modelini sıfırdan eğitmek aylar süren ve büyük veri gerektiren bir iş. Bu yüzden tanıma kısmında hazır ve açık kaynak bir model (faster-whisper) kullanıldı. Modele girmeden önceki tüm adımlar ise öğrenme amacıyla **elle yazıldı**:

- Bir ses dosyası bilgisayarda nasıl temsil edilir: örnekleme hızı, kanallar, genlik
- Örnekleme hızı neden ve nasıl değiştirilir, aliasing nedir
- Dijital filtreler sesi nasıl temizler
- Fourier dönüşümüyle ses frekanslarına nasıl ayrılır, gürültü frekans uzayında nasıl bastırılır
- Bir kayıtta konuşma olan ve olmayan yerler nasıl ayırt edilir
- Bir konuşma tanıma modeli sesi nasıl "görür": log-mel spektrogram

Böylece hem gerçekten kullanılabilen bir araç ortaya çıktı, hem de modern konuşma tanıma sistemlerinin önündeki ses işleme hattı adım adım anlaşılmış oldu.

## Özellikler

- **Her formatı okur:** mp3, wav, m4a, ogg, flac, hatta mp4 videoların ses kanalı
- **Ses ön işleme:** mono'ya indirme, 16 kHz'e yeniden örnekleme, yüksek geçiren filtre, spektral gürültü azaltma, normalizasyon
- **Konuşma tespiti (VAD):** Sessiz bölümler atlanır. Hem işlem hızlanır, hem de modelin sessizlikte metin uydurma riski azalır
- **GPU desteği:** NVIDIA ekran kartı varsa otomatik kullanılır, CUDA Toolkit kurmaya gerek yoktur. GPU yoksa CPU'ya geçer
- **Dil algılama:** Dil otomatik algılanır ya da elle seçilebilir (Türkçe, İngilizce ve Whisper'ın desteklediği diğer diller)
- **Çıktılar:** `.txt` (düz metin), `.srt` (video altyazısı), `.json` (zaman damgalı veri)
- **Görselleştirme:** dalga formu, tespit edilen konuşma bölgeleri ve log-mel spektrogram grafiği
- **Web arayüzü:** dosya yükleme veya mikrofonla kayıt, temizlenmiş sesi dinleme, sonuçları indirme

## Nasıl çalışır?

```
ses dosyası
  → yükleme (PyAV)
  → mono'ya indirme
  → 16 kHz'e yeniden örnekleme
  → yüksek geçiren filtre (80 Hz)
  → spektral gürültü azaltma
  → normalizasyon (-1 dBFS)
  → VAD: konuşma bölgelerinin tespiti
  → Whisper: yazıya dökme
  → txt / srt / json
```

| Adım | Ne yapar | Neden gerekli |
|---|---|---|
| **Yükleme** | Dosyayı açar, örnekleri float'a çevirir, kanalların ortalamasını alır | Her format sesi farklı saklar, hepsini ortak bir biçime getirmek gerekir |
| **Yeniden örnekleme** | 44.1 / 48 kHz sesi 16 kHz'e çevirir (polifaz filtre) | Whisper 16 kHz ses bekler. Filtre aliasing'i önler |
| **Yüksek geçiren filtre** | 80 Hz altını keser (4. dereceden Butterworth, ileri-geri uygulanır) | Elektrik uğultusu, rüzgar ve titreşim gibi konuşma dışı sesleri temizler |
| **Gürültü azaltma** | STFT ile frekanslara ayırır, en sessiz anlardan gürültü profili çıkarır, bu profile yakın enerjiyi bastırır | Fan, uğultu, hışırtı gibi sürekli gürültüleri azaltır |
| **Normalizasyon** | En yüksek genliği -1 dBFS'ye getirir | Kısık ve yüksek kayıtları aynı seviyeye çeker |
| **VAD** | 10 ms aralıklarla enerjiyi ölçer, arka plan seviyesinin 12 dB üstünü konuşma sayar | Sessizlikler işlenmez. Kısa duraklamalar birleştirilir, böylece model cümle bağlamını kaybetmez |
| **Whisper** | Konuşma bölgelerini metne çevirir (beam search, beam_size=5) | Asıl konuşma tanıma adımı |

Log-mel spektrogram, Whisper'ın kullandığı ayarlarla sıfırdan hesaplanır: 25 ms pencere, 10 ms adım, 80 mel bandı. Böylece modele giden girdi grafik üzerinde görülebilir.

## Performans

NVIDIA RTX 4060 Laptop GPU ve `turbo` (large-v3-turbo) modeliyle, 17 saniyelik gürültülü bir kayıt yaklaşık **0.8 saniyede** işlendi. Bu, gerçek zamanın yaklaşık **20 katı** hız demek.

## Kurulum

Python 3.10–3.12 gerekir.

```bash
git clone https://github.com/Ervanurb/ses-yazi.git
cd ses-yazi
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

- NVIDIA ekran kartı varsa model otomatik olarak GPU'da çalışır. Gerekli NVIDIA kütüphaneleri pip ile kurulur, ayrıca CUDA Toolkit kurmaya gerek yoktur.
- GPU yoksa CPU'da `small` modeli kullanılır.
- Modeller ilk çalıştırmada Hugging Face'ten indirilir: turbo yaklaşık 1.6 GB, small yaklaşık 500 MB.

## Kullanım

### Web arayüzü

```bash
.venv\Scripts\python app.py
```

Tarayıcıda http://127.0.0.1:7860 adresini açın. Dosya yükleyebilir ya da mikrofonla kayıt yapabilirsiniz. "Ses analizi" bölümünde Whisper'a giden temizlenmiş sesi dinleyebilir ve analiz grafiğini görebilirsiniz.

### Komut satırı

```bash
# Tek dosya: output/ klasörüne txt, srt ve json üretir
.venv\Scripts\python transcribe.py kayit.mp3

# Birden fazla dosya, dili sabitle, analiz grafiği de kaydet
.venv\Scripts\python transcribe.py a.mp3 b.m4a --dil tr --grafik

# Belirli bir model ve cihaz
.venv\Scripts\python transcribe.py kayit.mp3 --model small --cihaz cpu
```

| Seçenek | Açıklama |
|---|---|
| `--model` | `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo` (varsayılan: GPU'da turbo, CPU'da small) |
| `--dil` | Dil kodu, ör. `tr`, `en`. Boş bırakılırsa otomatik algılanır |
| `--cihaz` | `auto`, `cuda`, `cpu` |
| `--no-denoise` | Gürültü azaltmayı kapatır |
| `--no-vad` | Konuşma tespitini kapatır, tüm ses işlenir |
| `--grafik` | Analiz grafiğini de kaydeder |
| `--cikti` | Çıktı klasörü (varsayılan: `output`) |

### Sadece ses analizi

Whisper'ı çalıştırmadan, yalnızca ön işlemenin sonucunu görmek için:

```bash
.venv\Scripts\python analyze.py kayit.mp3
```

Bu komut konuşma bölümlerini ekrana yazar, `output/` klasörüne temizlenmiş wav dosyasını ve analiz grafiğini kaydeder.

## Proje yapısı

```
ses-yazi/
├── ses_yazi/
│   ├── audio.py        # yükleme, yeniden örnekleme, filtre, gürültü azaltma, VAD
│   ├── viz.py          # mel filtre bankası, log-mel spektrogram, grafikler
│   └── transcribe.py   # model yükleme (GPU/CPU), yazıya dökme, txt/srt/json
├── app.py              # Gradio web arayüzü
├── transcribe.py       # komut satırı: yazıya dökme
├── analyze.py          # komut satırı: sadece ses analizi
├── scripts/
│   ├── tts_ornek.ps1   # Windows'un Türkçe sesiyle test kaydı üretir
│   └── gurultu_ekle.py # kayda uğultu, hışırtı ve sessizlik ekler
├── samples/            # örnek test kayıtları
└── requirements.txt
```

## Geliştirme sürecinde öğrenilenler

- **VAD'ı fazla hassas yapmak doğruluğu düşürür.** İlk sürümde ses her 0.3 saniyelik duraklamada bölünüyordu. Whisper her parçayı ayrı gördüğü için cümle bağlamını kaybediyordu. 1 saniyeden kısa duraklamalar birleştirilince doğruluk belirgin şekilde arttı. Uzun sessizlikler yine atlanıyor.
- **Modelin doğruluğu ses kalitesinden çok sesin doğallığına bağlı.** Windows'un yapay "Tolga" sesiyle üretilen temiz kayıtta bile Whisper hata yaptı. En büyük model (large-v3) de aynı sonucu verdi. Model gerçek insan sesleriyle eğitildiği için robotik sesleri zor tanıyor.
- **Spektral gürültü azaltma sınırlıdır.** Sürekli gürültülerde (fan, uğultu) iyi çalışır, ani seslerde (kapı çarpması, köpek havlaması) etkisi azdır. Gürültüyü tamamen silmek yerine 10'da 1'ine indirmek, sesin metalik duyulmasını önler.

## Sınırlamalar

- `samples/` içindeki kayıtlar yapay sesle üretildi. Gerçek insan sesinde doğruluk çok daha yüksektir.
- Kimin konuştuğunu ayırt etmez (konuşmacı ayrımı yok).
- Canlı (gerçek zamanlı) yazıya dökme yoktur, kayıt bittikten sonra işlenir.
- Web arayüzü yerel kullanım için tasarlandı. Şifre koruması yoktur, internete açılmamalıdır.

## Gelecek planları

- Mikrofondan canlı yazıya dökme
- Konuşmacı ayrımı ("Konuşmacı 1", "Konuşmacı 2")
- Zamanla güncellenen gürültü profili ya da yapay zekâ tabanlı gürültü temizleme ile karşılaştırma
- Hugging Face Spaces üzerinde herkese açık demo

## Geliştirme notu

Bu proje geliştirilirken yapay zekâ asistanı **Claude**'dan (Anthropic) yararlanıldı. Projenin kapsamı ve yönü, adımların planlanması ve hangi özelliklerin ekleneceği kararları bana aitti. Claude bu doğrultuda kodun yazılmasında, hataların ayıklanmasında, test edilmesinde ve kullanılan ses işleme kavramlarının adım adım açıklanmasında yardımcı oldu.

## Kullanılan teknolojiler

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [PyAV](https://github.com/PyAV-Org/PyAV) · [NumPy](https://numpy.org/) · [SciPy](https://scipy.org/) · [Matplotlib](https://matplotlib.org/) · [Gradio](https://www.gradio.app/)
