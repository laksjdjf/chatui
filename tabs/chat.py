import gradio as gr
from tabs.setting import generate, get_prompt_from_history

stop_generate = False

def stop():
    global stop_generate
    stop_generate = True

def chat_handler(history, user=None, chatbot_beginning=None):
    global stop_generate
    stop_generate = False
    prompt = get_prompt_from_history(history, user, chatbot_beginning)

    if user is None:
        user, chatbot = history[-1]
        history = history[:-1]
    else:
        chatbot = chatbot_beginning

    for text in generate(prompt):
        if stop_generate:
            break

        chatbot += text
        yield "", history + [(user, chatbot)], gr.update(visible=False), gr.update(visible=True)

    yield "", history + [(user, chatbot)], gr.update(visible=True), gr.update(visible=False)

def undo_history(history):
    pre_message = history[-1][0] if history else ""
    return pre_message, history[:-1]

def view(history):
    return "\n\n".join([f'<div style="color: navy">user:{user.replace("\n", "")}</div>\n\n<div style="color: maroon">chatbot:{chatbot.replace("\n", "")}</div>' for user, chatbot in history])

def chat(user_avatar=None, chatbot_avatar=None):
    with gr.Blocks() as chat_interface:
        gr.Markdown("Chat用タブです。Shift+Enterでも送信できます。")
        chatbot = gr.Chatbot(avatar_images=[user_avatar, chatbot_avatar], layout="panel")
        user = gr.Textbox(label="User message", placeholder="Enter your message here...", lines=3)
        chatbot_beginning = gr.Textbox(label="Chatbot beginning", placeholder="Enter chatbot's first message here...", lines=2)

        generate_button = gr.Button("Generate", variant="primary")
        stop_button = gr.Button("Stop", visible=False)

        with gr.Row():
            continue_button = gr.Button("Continue")
            undo_button_chat = gr.Button("Undo")
            clear_button = gr.ClearButton([user, chatbot, chatbot_beginning])

        view_button = gr.Button("View", variant="secondary")
        view_text = gr.Markdown("")
        
        user.submit(chat_handler, inputs=[chatbot, user, chatbot_beginning], outputs=[user, chatbot, generate_button, stop_button])
        generate_button.click(chat_handler, inputs=[chatbot, user, chatbot_beginning], outputs=[user, chatbot, generate_button, stop_button])
        stop_button.click(stop)

        continue_button.click(chat_handler, inputs=[chatbot], outputs=[user, chatbot, generate_button, stop_button])
        undo_button_chat.click(undo_history, inputs=[chatbot], outputs=[user, chatbot])

        view_button.click(view, inputs=[chatbot], outputs=[view_text])

    return chat_interface