"""Tarayıcı arayüzü: dosya yükle veya mikrofondan kaydet, yazıya dök.

Çalıştır: python app.py   ->  http://127.0.0.1:7860
"""

import json
import tempfile
from pathlib import Path

import gradio as gr

from ses_yazi.audio import WHISPER_SR, detect_speech, preprocess, save_wav
from ses_yazi.transcribe import load_model, to_dict, to_srt, to_txt, transcribe
from ses_yazi.viz import plot_analysis

print("Model yükleniyor...")
MODEL, DEVICE = load_model()
print(f"Cihaz: {DEVICE}")

LANGUAGES = {"Otomatik": None, "Türkçe": "tr", "İngilizce": "en", "Almanca": "de", "Fransızca": "fr"}


def run(audio_path, language, denoise, vad):
    if not audio_path:
        raise gr.Error("Önce bir ses dosyası yükleyin veya kayıt yapın.")

    original, audio = preprocess(audio_path, denoise=denoise)
    speech = detect_speech(audio, WHISPER_SR, min_silence=1.0) if vad else None
    result = transcribe(MODEL, audio, speech, language=LANGUAGES[language])

    out = Path(tempfile.mkdtemp(prefix="ses_yazi_"))
    stem = Path(audio_path).stem
    srt_path = out / f"{stem}.srt"
    txt_path = out / f"{stem}.txt"
    json_path = out / f"{stem}.json"
    srt_path.write_text(to_srt(result), encoding="utf-8")
    txt_path.write_text(to_txt(result), encoding="utf-8")
    json_path.write_text(json.dumps(to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")

    clean_path = out / f"{stem}_temiz.wav"
    save_wav(clean_path, audio)
    png_path = out / f"{stem}_analiz.png"
    plot_analysis(original, audio, speech or [], png_path, title=Path(audio_path).name)

    duration = len(audio) / WHISPER_SR
    info = (
        f"**Dil:** {result.language} ({result.language_prob:.0%}) · **Süre:** {duration:.1f} s · "
        f"**İşlem:** {result.seconds:.1f} s ({duration / max(result.seconds, 1e-3):.1f}x gerçek zaman) · "
        f"**Cihaz:** {DEVICE}"
    )
    return (
        to_txt(result, timestamps=True),
        info,
        str(clean_path),
        str(png_path),
        [str(txt_path), str(srt_path), str(json_path)],
    )


with gr.Blocks(title="Ses → Yazı") as demo:
    gr.Markdown("# Ses → Yazı\nBir dosya yükleyin ya da mikrofonla kaydedin; ses temizlenir, konuşma bulunur ve Whisper ile yazıya dökülür.")
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Ses")
            language = gr.Dropdown(list(LANGUAGES), value="Türkçe", label="Dil")
            denoise = gr.Checkbox(value=True, label="Gürültü azaltma")
            vad = gr.Checkbox(value=True, label="Sessizlikleri atla (VAD)")
            button = gr.Button("Yazıya dök", variant="primary")
        with gr.Column(scale=2):
            text_out = gr.Textbox(label="Metin", lines=12)
            info_out = gr.Markdown()
            files_out = gr.File(label="İndir (txt / srt / json)", file_count="multiple")
    with gr.Accordion("Ses analizi", open=False):
        clean_out = gr.Audio(label="Temizlenmiş ses (Whisper'a giden)")
        plot_out = gr.Image(label="Dalga formu, konuşma bölgeleri, mel spektrogram")

    button.click(
        run,
        inputs=[audio_in, language, denoise, vad],
        outputs=[text_out, info_out, clean_out, plot_out, files_out],
    )

if __name__ == "__main__":
    demo.launch()
