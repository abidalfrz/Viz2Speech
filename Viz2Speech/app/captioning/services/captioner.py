import torch
from torch.cuda.amp import autocast
from .config import qwen_model, qwen_tokenizer
from unsloth import FastVisionModel
import io
from PIL import Image
import re

class ImageCaptioner:
    def __init__(self, qwen_model, qwen_tokenizer, device):
        self.qwen_model = qwen_model
        self.qwen_tokenizer = qwen_tokenizer
        self.device = device
    
    def compress_image(self, image, max_size=720, quality=85):

        image.thumbnail((max_size, max_size), Image.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")

        print(f"Compressed size: {compressed.size}")
        return compressed
    
    def extract_caption(self, raw_output):
        match = re.search(f'<CAPTION>(.*?)</CAPTION>', raw_output, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw_output.strip()
    
    def generate_caption(self, image, max_size=720, max_new_tokens=1024):
        FastVisionModel.for_inference(self.qwen_model)
        image = self.compress_image(image, max_size=max_size)
        REASONING_START = "<THINKING>"
        REASONING_END   = "</THINKING>"
        CAPTION_START   = "<CAPTION>"
        CAPTION_END     = "</CAPTION>"
        instruction = (
            "Deskripsikan gambar ini secara objektif untuk tunanetra. "
            f"Pertama, tuliskan analisis visual Anda di antara {REASONING_START} dan {REASONING_END}. "
            f"Setelah itu, berikan deskripsi akhir yang objektif, jelas,lengkap, dan siap dibacakan "
            f"oleh mesin TTS di antara {CAPTION_START} dan {CAPTION_END} dalam satu hingga dua kalimat."
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]},
        ]

        inputs_text = self.qwen_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            tokenize = False
        )

        inputs = self.qwen_tokenizer(
            image,
            inputs_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to(self.device)

        with torch.inference_mode():
            with autocast():
                generated_ids = self.qwen_model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=False, 
                    num_beams=5,   
                    repetition_penalty=1.5,       
                    no_repeat_ngram_size=4,  
                )
        
        input_len = inputs.input_ids.shape[1]
        generated_new_ids = generated_ids[:, input_len:]
        
        refined_caption = self.qwen_tokenizer.batch_decode(generated_new_ids, skip_special_tokens=True)[0]
        final_caption = self.extract_caption(refined_caption)
        return final_caption.strip() if final_caption.strip() != "" else "unk"
    
captioner = ImageCaptioner(qwen_model, qwen_tokenizer, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))