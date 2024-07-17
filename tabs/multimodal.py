import gradio as gr
from modules.llama_process import generate_chat
import base64
import re

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

def path2html(path):
    return f"<img src='/file={path}' style='max-width:200px; max-height:200px'></img>"

def html2path(html):
    match = re.search(r"src='/file=([^']+)'", html)
    if match:
        return match.group(1)
    return None
    
def echo(message, history, system):
    image = None
    messages = [{"role": "system", "content": system}]
    for user, assistant in history:
        if assistant is None:
            image = image_to_base64_data_uri(html2path(user))
        else:
            if image:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type" : "text", "text": user},
                        {"type": "image_url", "image_url": {"url": image }}
                    ]
                })
                image = None
            else:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type" : "text", "text": user}
                    ]
                })
            messages.append({"role": "assistant", "content": assistant})

    if len(message["files"])>0:
        image = image_to_base64_data_uri(message["files"][0])
        messages.append({
            "role": "user",
            "content": [
                {"type" : "text", "text": message["text"]},
                {"type": "image_url", "image_url": {"url": image } }
            ]
        })
        history += [(path2html(message["files"][0]), None)]
    else:
        messages.append({
            "role": "user",
            "content": [
                {"type" : "text", "text": message["text"]}
            ]
        })

    content = ""
    yield gr.update(value={"text": "", "files":[]}), history + [(message["text"], content)]
    
    for text in generate_chat(messages):
        content += text
        yield gr.update(interactive=True), history + [(message["text"], content)]

def undo_history(history):
    if len(history)>1:
        pre_message = history[-1][0]
        if history[-2][1] is None:
            pre_image = [html2path(history[-2][0])]
            pre_history = history[:-2]
        else:
            pre_image = []
            pre_history = history[:-1]
    elif len(history)==1:
        pre_message = history[-1][0]
        pre_image = []
        pre_history = []
    else:
        pre_message = ""
        pre_image = []
        pre_history = []
    return {"text": pre_message, "files": pre_image}, pre_history
    

def multimodal():
    with gr.Blocks() as multimodal_interface:
        chatbot = gr.Chatbot(height=600)
        chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False, lines=2)
        system = gr.Textbox(label="system", value="あなたは優秀なアシスタントです。与えられた画像について説明してください。", lines=2)

        with gr.Row():
            undo_button_chat = gr.Button("Undo")
            clear_button = gr.ClearButton([chat_input, chatbot])

        chat_input.submit(
            echo,
            inputs=[chat_input, chatbot, system],
            outputs=[chat_input, chatbot]
        )

        undo_button_chat.click(
            undo_history,
            inputs=[chatbot],
            outputs=[chat_input, chatbot]
        )

    return multimodal_interface
        
