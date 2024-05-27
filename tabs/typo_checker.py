import gradio as gr
from tabs.setting import eval_text, get_prompt
import pandas as pd

def typo_check_handler(text, threshold):

    perplexity, typo, text_splited = eval_text(text, threshold)

    texts = ""
    for b, text in zip(typo, text_splited):
        if b:
            texts += f"<span style='color:red'>{text}</span>"
        else:
            texts += f"{text}"

    return perplexity, texts

def typo_checker():
    with gr.Blocks() as typo_checker_interface:
        gr.Markdown("テキスト評価用タブです。")

        with gr.Row():
            with gr.Column():
                textbox = gr.Textbox(
                    label="Input",
                    placeholder="Enter your text here...",
                    interactive=True,
                    elem_classes=["prompt"],
                    lines=3,
                )
                
                threshold = gr.Slider(label="Threshold", minimum=0, maximum=1, step=0.001, value=0.1)
                eval_button = gr.Button("Evaluation", variant="primary")

            with gr.Column():
                perplexity = gr.Textbox(label=f"Perplexity", interactive=False,lines=1)
                output = gr.Markdown("")
        

        eval_button.click(
            typo_check_handler,
            inputs=[textbox, threshold],
            outputs=[perplexity, output]
        )
    return typo_checker_interface