import gradio as gr
from dotenv import load_dotenv
from src.chatbot import MultilingualChatbot, ChatConfig

load_dotenv()  # đọc file .env

cfg = ChatConfig(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens=512,
    temperature=0.7,
)
bot = MultilingualChatbot(cfg)

def stream_response(message: str, history: list):
    partial = ""
    for token in bot.stream(message):
        partial += token
        yield partial

def get_stats() -> str:
    s = bot.get_stats()
    langs = ", ".join(f"{k}:{v}" for k, v in s["lang_counts"].items())
    return f"Turns: {s['total_turns']}  |  Langs: {langs or 'none yet'}"

with gr.Blocks(title="Multilingual Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Chatbot Đa Ngôn Ngữ — VI / EN\nTự động nhận diện và trả lời đúng ngôn ngữ.")
    gr.ChatInterface(
        fn=stream_response,
        examples=[
            ["Xin chào! Giải thích AI cho tôi"],
            ["What is machine learning?"],
            ["Viết function Python tính fibonacci"],
        ],
        cache_examples=False,
    )
    with gr.Row():
        stats = gr.Textbox(label="Stats", value=get_stats(), interactive=False, scale=4)
        gr.Button("Refresh", scale=1).click(get_stats, outputs=stats)
        gr.Button("Reset", variant="stop", scale=1).click(lambda: (bot.reset(), get_stats()), outputs=stats)

demo.launch(server_name="0.0.0.0", server_port=7860)
