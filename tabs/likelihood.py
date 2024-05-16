import gradio as gr
from tabs.setting import eval, get_prompt
import pandas as pd

def likelihood_handler(prompt, *outputs):
    global pre_prompt, stop_generate

    result = {}
    log_likelihoods = []

    for output in outputs:
        if output:
            likelihood, log_likelihood = eval(prompt, output)
            result[output] = (likelihood)
            log_likelihoods.append(log_likelihood)

    df = pd.DataFrame(
        {"output": list(result.keys()), "likelihood": list(result.values()), "log_likelihood": log_likelihoods}
    )

    return df, result

def update_prompt(user, post_prompt=None):
    prompt = get_prompt(user, post_prompt)
    return gr.update(value=prompt, autoscroll=True)

def likelihood():
    with gr.Blocks() as likelihood_interface:
        gr.Markdown("テキスト評価用タブです。")

        with gr.Row():
            with gr.Column():
                prompt_textbox = gr.Textbox(
                    label="Input/Output",
                    placeholder="Enter your prompt here...",
                    interactive=True,
                    elem_classes=["prompt"],
                    lines=3,
                )
                
                user_textbox = gr.Textbox(label="user", value="こんにちんぽ", lines=2)
                default_button = gr.Button("Default")

                eval_button = gr.Button("Evaluation", variant="primary")

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
                    headers=["output", "likelihood", "log_likelihood"],
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