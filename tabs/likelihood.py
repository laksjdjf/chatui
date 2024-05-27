import gradio as gr
from tabs.setting import eval_output, get_prompt
import pandas as pd

def likelihood_handler(prompt, *outputs):

    outputs = [output for output in outputs if output] # remove empty outputs
    likelihoods, log_likelihoods, num_tokens = eval_output(prompt, outputs)

    df = pd.DataFrame(
        {"output": outputs, "likelihood": likelihoods, "log_likelihood": log_likelihoods, "num_tokens": num_tokens}
    )

    result = {output: likelihood for output, likelihood in zip(outputs, likelihoods)}

    return df.round(3), result

def update_prompt(user, post_prompt=None):
    prompt = get_prompt(user, post_prompt)
    return gr.update(value=prompt, autoscroll=True)

def likelihood():
    with gr.Blocks() as likelihood_interface:
        gr.Markdown("テキスト評価用タブです。")

        with gr.Row():
            with gr.Column():
                prompt_textbox = gr.Textbox(
                    label="Input",
                    placeholder="Enter your prompt here...",
                    interactive=True,
                    elem_classes=["prompt"],
                    lines=3,
                )
                
                
                default_button = gr.Button("Default")
                eval_button = gr.Button("Evaluation", variant="primary")

                user_textbox = gr.Textbox(label="input for default button", value="こんにちんぽ", lines=2)
                with gr.Group():
                    targets = [
                        gr.Textbox(
                            label=f"Target_{i}",
                            interactive=True,
                            lines=2,
                        )
                        for i in range(5)
                    ]

            with gr.Column():
                dataframe = gr.Dataframe(
                    headers=["output", "likelihood", "log_likelihood", "num_tokens"],
                    datatype=["str", "number", "number"],
                    row_count=5,
                )
                label = gr.Label("Result")
        

        eval_button.click(
            likelihood_handler,
            inputs=[prompt_textbox] + targets,
            outputs=[dataframe, label]
        )

        default_button.click(
            update_prompt,
            inputs=[user_textbox],
            outputs=[prompt_textbox],
        )

    return likelihood_interface