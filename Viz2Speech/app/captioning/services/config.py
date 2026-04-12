import torch
import os
from unsloth import FastVisionModel

QWEN_GRPO_PATH = os.path.join(os.path.dirname(__file__), "../artifacts/qwen-3vl-grpo-rl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

qwen_model, qwen_tokenizer = FastVisionModel.from_pretrained(
    QWEN_GRPO_PATH,
    load_in_4bit=True,
    use_gradient_checkpointing=True
)

