import gradio as gr
from modules.llama_process import eval_text

def eval_sentence_handler(text, relative, threshold):
    perplexity, result, text_splited = eval_text(text, relative)
    texts = ""
    for i, text in enumerate(text_splited):
        texts += get_color(text, result[i], threshold)

    return perplexity, texts

def get_color(text, prob, threshold):
    colors = ["#0000FF", "#00BFFF", "#32CD32", "#FFA500", "#FF0000"]
    thresholds = [float(t.strip()) for t in threshold.split(",")]
    color = [color for threshold, color in zip(thresholds, colors) if prob >= threshold][-1]
    return f'<span style="color:{color}">{text}</span>'

def eval_sentence():
    with gr.Blocks() as eval_sentence_interface:
        gr.Markdown("テキスト評価用タブです。")

        with gr.Row():
            with gr.Column():
                textbox = gr.Textbox(
                    label="Input",
                    placeholder="Enter your text here...",
                    interactive=True,
                    elem_classes=["prompt"],
                    lines=6,
                )
                
                relative = gr.Checkbox(label="Relative", value=False)
                thresholds = gr.Textbox("0.0, 0.2, 0.4, 0.6, 0.8", label="Thresholds", lines=1)
                eval_button = gr.Button("Evaluation", variant="primary")

            with gr.Column():
                perplexity = gr.Textbox(label=f"Perplexity", interactive=False,lines=1)
                output = gr.Markdown(label="Output")
        

        eval_button.click(
            eval_sentence_handler,
            inputs=[textbox, relative, thresholds],
            outputs=[perplexity, output]
        )
    return eval_sentence_interface