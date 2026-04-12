import torch
from chatterbox.tts import ChatterboxTTS
from safetensors.torch import load_file
import os

_original_load = torch.load

def _patched_load(*args, **kwargs):
    kwargs['map_location'] = 'cpu'
    return _original_load(*args, **kwargs)

torch.load = _patched_load


VOICE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../artifacts/chatterbox_id/model_t3.safetensors")

chatterbox_model = ChatterboxTTS.from_pretrained(device="cpu")
t3_state = load_file(VOICE_MODEL_PATH)
chatterbox_model.t3.load_state_dict(t3_state)