import gradio as gr
from modules.llama_process import generate, get_prompt_from_messages
from modules.utils import view

stop_generate = False
def stop():
    global stop_generate
    stop_generate = True

def chat_handler(history, system, user=None, assistant_beginning=""):
    global stop_generate
    stop_generate = False

    if user is None: # continue
        user, assistant = history[-1]
        history = history[:-1]
    else:
        assistant = assistant_beginning

    messages = [{"role": "system", "content": system}] if system else []
    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
    messages.append({"role": "user", "content": user})
    input_message = {"role": "assistant", "content": assistant}

    prompt = get_prompt_from_messages(messages, input_message, add_system=False)
    
    # input, chatbot, generate_button, stop_button
    yield gr.update(value=""), history + [(user, assistant)], gr.update(visible=False), gr.update(visible=True)

    for text in generate(prompt):
        if stop_generate:
            break
        assistant += text
        yield gr.update(interactive=True), history + [(user, assistant)], gr.update(visible=False), gr.update(visible=True)

    yield gr.update(interactive=True), history + [(user, assistant)], gr.update(visible=True), gr.update(visible=False)

def generate_input_handler(history, system, user):
    global stop_generate
    stop_generate = False

    messages = [{"role": "system", "content": system}] if system else []
    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
    input_message = {"role": "user", "content": user}

    prompt = get_prompt_from_messages(messages, input_message, add_system=False)

    texts = user
    for text in generate(prompt):
        texts += text
        if stop_generate:
            break
        yield texts, gr.update(visible=False), gr.update(visible=True)
    yield texts, gr.update(visible=True), gr.update(visible=False)

def undo_history(history):
    pre_message = history[-1][0] if history else ""
    return pre_message, history[:-1]

def apply_icon(user_icon, assistant_icon):
    return gr.update(avatar_images=[user_icon, assistant_icon])

def chat():
    with gr.Blocks() as chat_interface:
        gr.Markdown("Chat用タブです。Shift+Enterでも送信できます。")
        chatbot = gr.Chatbot(height=600)
        user = gr.Textbox(label="User message", placeholder="Enter your message here...", lines=3)
        assistant_beginning = gr.Textbox(label="Assistant beginning", placeholder="Enter assistant's first sentence here...", lines=2)

        generate_button = gr.Button("Generate", variant="primary")
        stop_button = gr.Button("Stop", visible=False)

        generate_input_button = gr.Button("Generate input by AI", variant="secondary")

        with gr.Row():
            continue_button = gr.Button("Continue")
            undo_button_chat = gr.Button("Undo")
            clear_button = gr.ClearButton([user, chatbot, assistant_beginning])

        with gr.Accordion("チャット設定"):
            system = gr.Textbox(label="system", value="あなたは優秀なアシスタントです。", lines=2)
            with gr.Row():
                with gr.Group():
                    user_name = gr.Textbox(label="User name", value="user", lines=1)
                    user_icon = gr.Image(label="User icon", type="filepath")
                with gr.Group():
                    assistant_name = gr.Textbox(label="Assistant name", value="assistant", lines=1)
                    assistant_icon = gr.Image(label="Assistant icon", type="filepath")
            icon_apply_button = gr.Button("Apply", variant="primary")

        view_button = gr.Button("View", variant="secondary")
        view_markdown = gr.Markdown("")
        
        user.submit(chat_handler, inputs=[chatbot, system, user, assistant_beginning], outputs=[user, chatbot, generate_button, stop_button])
        generate_button.click(chat_handler, inputs=[chatbot, system, user, assistant_beginning], outputs=[user, chatbot, generate_button, stop_button])
        stop_button.click(stop)

        generate_input_button.click(generate_input_handler, inputs=[chatbot, system, user], outputs=[user, generate_button, stop_button])

        continue_button.click(chat_handler, inputs=[chatbot, system], outputs=[user, chatbot, generate_button, stop_button])
        undo_button_chat.click(undo_history, inputs=[chatbot], outputs=[user, chatbot])

        icon_apply_button.click(apply_icon, inputs=[user_icon, assistant_icon], outputs=[chatbot])

        view_button.click(view, inputs=[chatbot, user_name, assistant_name, user_icon, assistant_icon, system], outputs=[view_markdown])

    return chat_interface