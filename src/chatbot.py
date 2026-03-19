import torch
from dataclasses import dataclass
from threading import Thread
from typing import Generator, Optional
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TextIteratorStreamer
)
from .lang_detect import detect_language, get_system_prompt

@dataclass
class ChatConfig:
    model_id: str         = "Qwen/Qwen2.5-1.5B-Instruct"
    max_history: int      = 10
    max_new_tokens: int   = 512
    temperature: float    = 0.7
    top_p: float          = 0.9
    repetition_penalty: float = 1.1 
    use_4bit: bool        = True

class MultilingualChatbot:
    def __init__(self, config: Optional[ChatConfig] = None):
        self.cfg     = config or ChatConfig()
        self.history = []
        self.stats   = {"total_turns": 0, "lang_counts": {}}
        self._load_model()

    def _load_model(self):
        print(f"Loading {self.cfg.model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        load_kwargs = {}
        if torch.cuda.is_available() and self.cfg.use_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs["device_map"] = "auto"
        elif torch.backends.mps.is_available():
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id, **load_kwargs
        )
        if torch.backends.mps.is_available() and not torch.cuda.is_available():
            self.model = self.model.to("mps")
        self.model.eval()
        print("Model loaded ✓")

    def _build_messages(self, user_msg: str) -> list:
        lang   = detect_language(user_msg)
        system = get_system_prompt(lang)
        self.stats["total_turns"] += 1
        self.stats["lang_counts"][lang] = self.stats["lang_counts"].get(lang, 0) + 1
        recent = self.history[-(self.cfg.max_history * 2):]
        return [{"role": "system", "content": system}] + recent + [{"role": "user", "content": user_msg}]

    def chat(self, user_msg: str) -> str:
        messages = self._build_messages(user_msg)
        text     = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs   = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                do_sample=True,
                repetition_penalty=self.cfg.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        self.history += [{"role": "user", "content": user_msg}, {"role": "assistant", "content": response}]
        return response
    
    def stream(self, user_msg: str) -> Generator:
        messages = self._build_messages(user_msg)
        text     = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs   = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        Thread(target=self.model.generate, kwargs={
            **inputs, "streamer": streamer,
            "max_new_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }).start()
        full = ""
        for token in streamer:
            full += token
            yield token
        self.history += [{"role": "user", "content": user_msg}, {"role": "assistant", "content": full}]

    def reset(self):
        self.history = []

    def get_stats(self) -> dict:
        return self.stats