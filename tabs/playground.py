import gradio as gr
from modules.llama_process import generate, get_prompt_from_messages
import ast

stop_generate = False
def stop():
    global stop_generate
    stop_generate = True

def get_message_from_history(history, system=None):
    messages = [{"role": "system", "content": system}] if system else []
    for user_message, assistant_message in history:
        if user_message is not None:
            messages.append({"role": "user", "content": user_message if user_message != "**empty**" else ""})
        else:
            messages.append({"role": "assistant", "content": assistant_message if assistant_message != "**empty**" else ""})
    return messages

def generate_handler(history, system):
    global stop_generate
    stop_generate = False

    messages = get_message_from_history(history, system)
    prompt = get_prompt_from_messages(messages[:-1], messages[-1], add_system=False)
    
    user, assistant = history[-1]
    user = user if user != "**empty**" else ""
    assistant = assistant if assistant != "**empty**" else ""
    history = history[:-1]

    yield history + [(user, assistant)], gr.update(visible=False), gr.update(visible=True)

    for text in generate(prompt):
        if stop_generate:
            break
        
        if user is None:
            assistant += text
        else:
            user += text
        
        yield history + [(user, assistant)], gr.update(visible=False), gr.update(visible=True)

    yield history + [(user, assistant)], gr.update(visible=True), gr.update(visible=False)

def delete_last_message(history):
    return history[:-1]

def add_user_handler(history, input_message):
    input_message = input_message if input_message != "" else "**empty**"
    return history + [(input_message, None)], ""

def add_assistant_handler(history, input_message):
    input_message = input_message if input_message != "" else "**empty**"
    return history + [(None, input_message)], ""

def update_dict_handler(history):
    return str(get_message_from_history(history))

def apply_dict_handler(dict_view):
    dict_list = ast.literal_eval(dict_view)
    history = []
    for i in range(len(dict_list)):
        content = dict_list[i]["content"] if dict_list[i]["content"] != "" else"**empty**"
        if dict_list[i]["role"] == "user":
            history.append((content, None))
        elif dict_list[i]["role"] == "assistant":
            history.append((None, content))

    return history


def playground():
    with gr.Blocks() as playground_interface:
        gr.Markdown("Playground用タブです。")
        system = gr.Textbox(label="system", value="あなたは優秀なアシスタントです。", lines=2)
        chatbot = gr.Chatbot(height=600)
        input_message = gr.Textbox(label="Input message", lines=3)

        with gr.Row():
            add_user_button = gr.Button("Add User")
            add_assistant_button = gr.Button("Add Assistant")

        generate_button = gr.Button("Generate", variant="primary")
        stop_button = gr.Button("Stop", visible=False)

        with gr.Row():
            delete_button_chat = gr.Button("Delete")
            clear_button = gr.ClearButton([input_message, chatbot])

        with gr.Row():
            update_dict_button = gr.Button("Update dict", variant="secondary")
            apply_dict_button = gr.Button("Apply dict", variant="primary")
        dict_view = gr.Textbox(label="dict", lines=10)

        add_user_button.click(add_user_handler, inputs=[chatbot, input_message], outputs=[chatbot, input_message])
        add_assistant_button.click(add_assistant_handler, inputs=[chatbot, input_message], outputs=[chatbot, input_message])
        
        generate_button.click(generate_handler, inputs=[chatbot, system], outputs=[chatbot, generate_button, stop_button])
        stop_button.click(stop)

        delete_button_chat.click(delete_last_message, inputs=[chatbot], outputs=[chatbot])

        update_dict_button.click(update_dict_handler, inputs=[chatbot], outputs=[dict_view])
        apply_dict_button.click(apply_dict_handler, inputs=[dict_view], outputs=[chatbot])

    return playground_interface