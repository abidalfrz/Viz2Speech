import gradio as gr
import requests
import tempfile
import os
import re
from num2words import num2words
from pyngrok import ngrok

CAPTIONING_API_URL = "http://127.0.0.1:8001/caption"
TTS_API_URL = "http://127.0.0.1:8000/generate"

def num_converter(match):
    num = int(match.group())
    return num2words(num, lang='id')

def image_to_speech(image_path, ref_audio_path, caption_mode):
    if image_path is None:
        raise gr.Error("Please upload an image first to get started.")

    try:
        with open(image_path, "rb") as img_file:
            mode_val = caption_mode.lower()
            resp_caption = requests.post(
                CAPTIONING_API_URL, 
                files={"image": img_file},
                data={"mode": mode_val}
            )
            
            if resp_caption.status_code != 200:
                raise Exception(f"Captioning API error: {resp_caption.status_code} - {resp_caption.text}")
            caption = resp_caption.json().get("caption", "")
    except Exception as e:
        raise gr.Error(f"Vision processing failed: {str(e)}")

    yield caption, gr.update(value=None)

    try:
        conv_caption = re.sub(r'\d+', num_converter, caption)
        if ref_audio_path is not None:
            with open(ref_audio_path, "rb") as audio_file:
                resp_tts = requests.post(
                    TTS_API_URL,
                    data={"text": conv_caption},
                    files={"ref_audio": audio_file}
                )
        else:
            resp_tts = requests.post(TTS_API_URL, data={"text": conv_caption})

        if resp_tts.status_code != 200:
            raise Exception(f"TTS API error: {resp_tts.status_code} - {resp_tts.text}")
            
        if not resp_tts.content:
            raise Exception("TTS API returned 0 bytes of audio data. Please check your TTS server.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(resp_tts.content)
            temp_audio.flush()
            os.fsync(temp_audio.fileno()) 
            temp_audio_path = temp_audio.name

    except Exception as e:
        raise gr.Error(f"Speech synthesis failed: {str(e)}")

    yield caption, temp_audio_path


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-main: #0B0F19;
    --panel-bg: rgba(17, 24, 39, 0.7);
    --panel-border: rgba(255, 255, 255, 0.08);
    --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    --accent-glow: rgba(99, 102, 241, 0.4);
    --text-main: #F9FAFB;
    --text-muted: #9CA3AF;
    --input-bg: rgba(31, 41, 55, 0.5);
    --radius-xl: 24px;
    --radius-md: 16px;
}

body, .gradio-container {
    background-color: var(--bg-main) !important;
    background-image: 
        radial-gradient(circle at 15% 50%, rgba(99, 102, 241, 0.08), transparent 25%),
        radial-gradient(circle at 85% 30%, rgba(168, 85, 247, 0.08), transparent 25%) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-main) !important;
}

.app-header { text-align: center; padding: 60px 0 40px; }

.app-badge {
    display: inline-block;
    font-family: 'Outfit', sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #fff;
    background: var(--accent-gradient);
    padding: 6px 16px;
    border-radius: 30px;
    margin-bottom: 24px;
    box-shadow: 0 4px 15px var(--accent-glow);
}

.app-title {
    font-family: 'Outfit', sans-serif;
    font-size: 64px;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0 0 16px;
    background: linear-gradient(to right, #ffffff 20%, #a855f7 40%, #818cf8 60%, #ffffff 80%);
    background-size: 200% auto;
    color: #000;
    background-clip: text;
    text-fill-color: transparent;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 4s linear infinite;
    text-shadow: 0px 4px 40px rgba(168, 85, 247, 0.4);
}

@keyframes shine { to { background-position: 200% center; } }

.app-subtitle {
    font-size: 18px;
    color: var(--text-muted);
    font-weight: 300;
    max-width: 550px;
    margin: 0 auto;
    line-height: 1.5;
}

.glass-panel {
    background: var(--panel-bg) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: var(--radius-xl) !important;
    padding: 32px !important;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2) !important;
}

.panel-header {
    font-family: 'Outfit', sans-serif;
    font-size: 20px;
    font-weight: 500;
    color: #fff;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.panel-step {
    background: rgba(255,255,255,0.1);
    color: #fff;
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    font-size: 14px;
    font-weight: 600;
}

.upload-zone {
    border: 2px dashed rgba(255,255,255,0.15) !important;
    border-radius: var(--radius-md) !important;
    background: var(--input-bg) !important;
    transition: all 0.3s ease !important;
}

.upload-zone:hover {
    border-color: #818cf8 !important;
    background: rgba(99, 102, 241, 0.05) !important;
}

.block label span, label > span, .gr-radio label span {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}

#output-text textarea, #output-audio {
    background: rgba(10, 15, 25, 0.6) !important;
    border: 1px solid rgba(168, 85, 247, 0.25) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: inset 0 6px 12px rgba(0,0,0,0.4) !important;
    padding: 16px !important;
    transition: all 0.3s ease !important;
}

#output-text textarea:hover, #output-audio:hover {
    border-color: rgba(168, 85, 247, 0.6) !important;
    box-shadow: inset 0 6px 12px rgba(0,0,0,0.4), 0 0 15px rgba(168, 85, 247, 0.15) !important;
}

#output-text textarea {
    color: #fff !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
}

#output-audio .audio-container {
    background: transparent !important;
    border: none !important;
}

#magic-btn {
    background: var(--accent-gradient) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 18px 24px !important;
    margin-top: 10px !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    width: 100% !important;
}

#magic-btn:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 25px var(--accent-glow) !important;
}

#magic-btn:active {
    transform: translateY(1px) scale(0.99) !important;
}

.hint-text {
    font-size: 12px;
    color: #6b7280;
    margin-top: 8px;
    display: block;
}

.app-footer {
    text-align: center;
    margin-top: 60px;
    padding-bottom: 40px;
    color: #4b5563;
    font-size: 14px;
}
"""

header_html = """
<div class="app-header">
    <div class="app-badge">Image-to-Speech Engine</div>
    <h1 class="app-title">Viz2Speech</h1>
    <p class="app-subtitle">A seamless pipeline translating visual data into Indonesian natural audio narratives using Optimized Vision-Language Models and Text-to-Speech architectures.</p>
</div>
"""

with gr.Blocks(
    title="Viz2Speech - Image to Indonesian Speech"
) as demo:

    gr.HTML(header_html)

    with gr.Column(elem_classes="glass-panel"):
        gr.HTML('<div class="panel-header"><span class="panel-step">1</span> Input Configuration</div>')

        with gr.Row(elem_classes="gap-row"):
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Target Image",
                    type="filepath",
                    elem_classes="upload-zone",
                )
            
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Reference Voice Cloning (Optional)",
                    type="filepath",
                    elem_classes="upload-zone",
                )
                gr.HTML('<span class="hint-text">Leave this blank to use the default system voice.</span>')

        gr.HTML('<div style="height: 16px;"></div>')

        caption_mode = gr.Radio(
            choices=["Fast", "Detailed"],
            value="Fast",
            label="Captioning Mode",
            info="Fast mode is optimized for speed. Detailed mode provides deeper analysis."
        )

        gr.HTML('<div style="height: 16px;"></div>')

        submit_button = gr.Button(
            "Generate Audio",
            elem_id="magic-btn",
        )

    gr.HTML('<div style="height: 32px;"></div>')

    with gr.Column(elem_classes="glass-panel"):
        gr.HTML('<div class="panel-header"><span class="panel-step">2</span> Generated Results</div>')

        with gr.Row(elem_classes="gap-row"):
            with gr.Column(scale=1):
                output_caption = gr.Textbox(
                    label="Generated Image Caption",
                    lines=6,
                    interactive=False,
                    placeholder="The image description will be generated here...",
                    elem_id="output-text" 
                )

            with gr.Column(scale=1):
                output_audio = gr.Audio(
                    label="Generated TTS Audio",
                    autoplay=True,
                    elem_id="output-audio"
                )

    gr.HTML('<div class="app-footer">2026 flaptzy. All rights reserved.</div>')
    def disable_btn():
        return gr.update(interactive=False, value="Processing...⏳")

    def enable_btn():
        return gr.update(interactive=True, value="Generate Audio")

    submit_button.click(
        fn=disable_btn,
        inputs=None,
        outputs=submit_button,
        queue=False
    ).then(
        fn=image_to_speech,
        inputs=[image_input, audio_input, caption_mode],
        outputs=[output_caption, output_audio],
    ).then(
        fn=enable_btn,
        inputs=None,
        outputs=submit_button,
        queue=False
    )

if __name__ == "__main__":
    ngrok.set_auth_token("YOUR_NGROK_TOKEN")
    public_url = ngrok.connect(7860)
    print(f"Public URL: {public_url}")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css=custom_css, theme=gr.themes.Base(font=gr.themes.GoogleFont("Inter")))
