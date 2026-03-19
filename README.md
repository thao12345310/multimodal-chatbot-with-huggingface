# multimodal-chatbot-with-huggingface
# Chatbot Đa Ngôn Ngữ (VI / EN)

Chatbot tự động nhận diện ngôn ngữ và trả lời bằng tiếng Việt hoặc English, xây dựng trên HuggingFace Transformers với khả năng fine-tuning bằng LoRA.

---

## Tính năng

- Tự động phát hiện ngôn ngữ (tiếng Việt, English, và nhiều ngôn ngữ khác)
- Trả lời đúng ngôn ngữ người dùng đang dùng
- Streaming response — hiện từng token như ChatGPT
- Lưu lịch sử hội thoại (multi-turn conversation)
- Giao diện web với Gradio
- Fine-tuning với LoRA trên dataset tiếng Việt
- Đánh giá chất lượng model với BLEU, ROUGE, BERTScore

---

## Cấu trúc project

```
multilingual-chatbot/
├── src/
│   ├── __init__.py        # File rỗng — để Python nhận src/ là package
│   ├── chatbot.py         # Class MultilingualChatbot (core)
│   ├── lang_detect.py     # Phát hiện ngôn ngữ + system prompts
│   └── metrics.py         # BLEU, ROUGE, BERTScore
├── data/
│   ├── train.jsonl        # Data fine-tuning (tạo bởi prepare_data.py)
│   └── valid.jsonl        # Data validation
├── outputs/               # LoRA adapters sau fine-tune
├── app.py                 # Entry point — Gradio web UI
├── prepare_data.py        # Tải và chuẩn bị dataset tiếng Việt
├── train.py               # Fine-tuning với LoRA
├── eval.py                # Đánh giá model
├── compare_results.py     # So sánh kết quả trước/sau fine-tune
├── verify_env.py          # Kiểm tra môi trường
├── test_cases.json        # Bộ câu hỏi test chuẩn
├── requirements.txt
├── .env                   # API keys (KHÔNG commit lên git)
└── .gitignore
```

---

## Yêu cầu hệ thống

| Thiết bị | RAM / VRAM | Model tối đa |
|---|---|---|
| Mac Apple Silicon (M1–M4) | 16GB | Qwen2.5-3B (4-bit) |
| Mac Apple Silicon (M1–M4) | 24GB+ | Qwen2.5-7B (4-bit) |
| NVIDIA GPU | 8GB VRAM | Qwen2.5-1.5B |
| NVIDIA GPU | 16GB VRAM | Qwen2.5-7B (4-bit) |
| Google Colab (T4 free) | 15GB VRAM | Qwen2.5-7B (4-bit) |
| CPU only | 8GB RAM | Qwen2.5-0.5B |

- Python 3.10 trở lên
- macOS 12.3+ (nếu dùng Apple Silicon)

---

## Cài đặt

### 1. Clone repo và tạo môi trường ảo

```bash
git clone https://github.com/YOUR_USERNAME/multilingual-chatbot.git
cd multilingual-chatbot

# Mac / Linux
python3.11 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 2. Cài thư viện

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Lưu ý NVIDIA GPU:** Cài PyTorch với đúng phiên bản CUDA trước:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 3. Tạo file `.env`

```bash
cp .env.example .env
# Mở .env và điền HuggingFace token
```

Nội dung file `.env`:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
# ANTHROPIC_API_KEY=sk-ant-xxx   # tuỳ chọn, dùng cho LLM-as-Judge
```

Lấy token tại: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 4. Kiểm tra môi trường

```bash
python verify_env.py
```

Output khi thành công:

```
=== Environment Check ===
  ✓ Python >= 3.10        3.11.9
  ✓ torch                 2.3.0
  ✓ transformers          4.44.2
  ✓ accelerate            0.31.0
  ✓ peft                  0.11.1
  ✓ gradio                4.40.0
  ✓ Apple MPS             Metal GPU ✓

All OK!
```

---

## Chạy ứng dụng

```bash
# Đảm bảo đã activate venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Khởi động web app
python app.py
```

Mở trình duyệt vào **http://localhost:7860**

Lần đầu chạy sẽ tải model (~1–3GB tùy model). Các lần sau load từ cache, mất khoảng 5–10 giây.

---

## Fine-tuning

### Bước 1 — Chuẩn bị data

```bash
python prepare_data.py
```

Tải dataset `bkai-foundation-models/vi-alpaca` (~50K mẫu tiếng Việt), lọc và tạo ra:
- `data/train.jsonl` — 7,600 mẫu
- `data/valid.jsonl` — 400 mẫu

### Bước 2 — Chạy fine-tuning

```bash
python train.py
```

Cấu hình mặc định: LoRA rank=16, 3 epochs, batch size=4. Thời gian chạy:
- NVIDIA T4 (Colab): ~2–3 giờ
- Mac M4 Air 16GB: không khuyến nghị (dùng `mlx_lm.lora` thay thế)

Kết quả lưu vào `./outputs/final/`

### Dùng model sau fine-tune

Thay `model_id` trong `app.py`:

```python
cfg = ChatConfig(model_id="./outputs/final")
```

---

## Đánh giá model

### Chạy đánh giá

```bash
# Trước fine-tune
python eval.py --model Qwen/Qwen2.5-1.5B-Instruct --output before.json

# Sau fine-tune
python eval.py --model ./outputs/final --output after.json

# So sánh
python compare_results.py before.json after.json
```

### Metrics được dùng

| Metric | Ý nghĩa | Tốt khi |
|---|---|---|
| `contains_pass` | Output có chứa từ khóa cần thiết | Càng cao càng tốt |
| `avg_bleu` | Độ trùng n-gram với đáp án chuẩn | Càng cao càng tốt |
| `avg_rouge1` | F1 overlap unigram | Càng cao càng tốt |
| `bertscore_f1` | Semantic similarity (0–1) | > 0.90 là tốt |
| `avg_latency_s` | Thời gian phản hồi (giây) | Càng thấp càng tốt |

### Kết quả mẫu

```
=== So sánh trước / sau fine-tune ===
Metric               Trước        Sau   Thay đổi
----------------------------------------------------
contains_pass            18         24     +6.000 ↑
avg_bleu              12.40      28.70    +16.300 ↑
avg_rouge1             0.341      0.612    +0.271 ↑
bertscore_f1           0.847      0.921    +0.074 ↑
avg_latency_s           2.14       2.31    +0.170 ↓
```

---

## Cấu hình model

Chỉnh trong `app.py`:

```python
cfg = ChatConfig(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",  # thay đổi model tại đây
    max_history=10,       # số lượt hội thoại giữ lại
    max_new_tokens=512,   # độ dài tối đa của response
    temperature=0.7,      # 0.1=xác định, 1.0=sáng tạo
    top_p=0.9,
    use_4bit=True,        # bật 4-bit quant (chỉ NVIDIA GPU)
)
```

### Các model được hỗ trợ

| Model | Kích thước | Tiếng Việt | RAM cần |
|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | Cơ bản | 2GB |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Tốt | 4GB |
| `Qwen/Qwen2.5-7B-Instruct` | 7B (4-bit) | Rất tốt | 8GB |
| `vilm/vinallama-7b-chat` | 7B | Xuất sắc (VI only) | 16GB |

---

## Xử lý lỗi thường gặp

**`ModuleNotFoundError: No module named 'transformers'`**

```bash
# Chưa activate venv
source venv/bin/activate
```

**`Error: from src.chatbot import ...`**

```bash
# Đang chạy sai thư mục
cd multilingual-chatbot
python app.py
```

**`RuntimeError: MPS backend out of memory`** (Mac)

```python
# Giảm max_new_tokens hoặc dùng model nhỏ hơn
cfg = ChatConfig(model_id="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=256)
```

**`bitsandbytes` lỗi trên Mac**

```bash
# Xóa dòng bitsandbytes khỏi requirements.txt — không dùng được trên MPS
# Dùng mlx-lm thay thế cho Apple Silicon
pip install mlx mlx-lm
```

---

## Tech stack

- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — load model, tokenizer, inference
- [PEFT](https://huggingface.co/docs/peft) — LoRA fine-tuning
- [TRL](https://huggingface.co/docs/trl) — SFTTrainer
- [MLX-LM](https://github.com/ml-explore/mlx-examples) — tối ưu cho Apple Silicon
- [Gradio](https://gradio.app) — web UI
- [langdetect](https://github.com/Mimino666/langdetect) — phát hiện ngôn ngữ
- [sacrebleu](https://github.com/mjpost/sacrebleu) + [rouge-score](https://github.com/google-research/google-research/tree/master/rouge) + [bert-score](https://github.com/Tiiiger/bert_score) — evaluation metrics

---

## License

MIT License — xem file [LICENSE](LICENSE) để biết thêm chi tiết.