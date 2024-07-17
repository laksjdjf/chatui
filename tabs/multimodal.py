import gradio as gr
from modules.llama_process import generate_chat
import base64

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"
    
def echo(message, history, system):
    image = None
    messages = [{"role": "system", "content": system}]
    for user, assistant in history:
        if user is None:
            image = image_to_base64_data_uri(assistant)
        else:
            if image:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type" : "text", "text": user},
                        {"type": "image_url", "image_url": {"url": image } }
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
    else:
        messages.append({
            "role": "user",
            "content": [
                {"type" : "text", "text": message["text"]}
            ]
        })

    content = ""
    for text in generate_chat(messages):
        content += text
        yield content

def multimodal():
    return gr.ChatInterface(
        fn=echo, 
        additional_inputs=[
            gr.Textbox(label="system", placeholder="Type your message here...", lines=2),
        ],
        multimodal=True
    )