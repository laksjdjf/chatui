import gradio as gr
from modules.llama_process import get_token_split

def tokenizer():
    with gr.Blocks() as tokenizer_interface:
        text = gr.Textbox(lines=5, label="Enter text")
        length = gr.Markdown("# Tokens 0   Characters 0")
        with gr.Row():
            splited = gr.Markdown("")
            ids = gr.Markdown("")

        text.change(get_token_split, inputs=[text], outputs=[splited, ids, length])

    return tokenizer_interface