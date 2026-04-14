# TTS/services/generator.py

import torch
import numpy as np
from .config import chatterbox_model
import os   

DEFAULT_REF = os.path.join(os.path.dirname(__file__), "default_voice.wav")

class VoiceGenerator:
    def __init__(self, model, device="cpu"):
        self.device = device
        self.model = model
        
    def generate_speech(self, text, ref_audio_path):
        with torch.inference_mode():
            if ref_audio_path is not None:
                wav_audio_clone = self.model.generate(
                    text,
                    audio_prompt_path=ref_audio_path
                )
            else:
                wav_audio_clone = self.model.generate(text, audio_prompt_path=DEFAULT_REF)
        
        audio_numpy = wav_audio_clone.squeeze().cpu().numpy()
        
        return self.model.sr, audio_numpy
    
generator = VoiceGenerator(chatterbox_model)