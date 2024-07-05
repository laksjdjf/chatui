import gradio as gr
from modules.llama_process import generate, get_prompt_from_messages, get_config

pre_prompt = ""
stop_generate = False

def stop():
    global stop_generate
    stop_generate = True

def completion_handler(prompt):
    global pre_prompt, stop_generate
    pre_prompt = prompt
    stop_generate = False

    for text in generate(prompt):
        if stop_generate:
            break

        prompt += text
        yield gr.update(value=prompt), gr.update(visible=False), gr.update(visible=True)

    yield gr.update(value=prompt), gr.update(visible=True), gr.update(visible=False)

def undo_prompt():
    return gr.update(value=pre_prompt)

def update_prompt(user, pre_prompt=None):
    messages = [{"role": "user", "content": user}]
    input_message = {"role": "assistant", "content": ""}
    prompt = get_prompt_from_messages(messages, input_message, add_system=pre_prompt is None)
    if pre_prompt:
        prompt = pre_prompt + get_config().assistant_suffix + prompt
    return gr.update(value=prompt, autoscroll=True)

def completion():
    with gr.Blocks() as completion_interface:
        gr.Markdown("Completion用タブです。Shift+Enterでも送信できます。")
        prompt_textbox = gr.Textbox(
            label="Input/Output",
            placeholder="Enter your prompt here...",
            interactive=True,
            lines=5,
        )

        generate_button = gr.Button("Generate", variant="primary")
        stop_button = gr.Button("Stop", visible=False)

        with gr.Row():
            undo_button = gr.Button("Undo")
            default_button = gr.Button("Default")
            add_button = gr.Button("Add Input")

        user_textbox = gr.Textbox(label="user", value="", lines=3)

        prompt_textbox.submit(
            completion_handler,
            inputs=[prompt_textbox],
            outputs=[prompt_textbox, generate_button, stop_button],
        )

        generate_button.click(
            completion_handler,
            inputs=[prompt_textbox],
            outputs=[prompt_textbox, generate_button, stop_button],
        )

        stop_button.click(stop)

        undo_button.click(
            undo_prompt,
            inputs=None,
            outputs=[prompt_textbox],
        )

        default_button.click(
            update_prompt,
            inputs=[user_textbox],
            outputs=[prompt_textbox],
        )

        add_button.click(
            update_prompt,
            inputs=[user_textbox, prompt_textbox],
            outputs=[prompt_textbox],
        )

    return completion_interface