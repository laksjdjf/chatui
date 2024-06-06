import gradio as gr
from tabs.setting import generate, get_prompt_from_history, view

stop_generate = False

def stop():
    global stop_generate
    stop_generate = True

def chat_handler(history_b, system_a, system_b, first_message, n_completion):
    global stop_generate
    stop_generate = False

    if len(history_b) > 0:
        chat_b = history_b[-1][1]
        history_a = [(first_message, history_b[0][0])] + [(history_b[i-1][1], history_b[i][0]) for i in range(1, len(history_b))]
    else:
        chat_b = first_message
        history_a = []

    for i in range(n_completion):
        chat_a = ""
        prompt = get_prompt_from_history(history_a, chat_b, system=system_a)
        
        if stop_generate:
            break

        for text in generate(prompt):
            if stop_generate:
                break

            chat_a += text
            yield history_b + [(chat_a, "")], gr.update(visible=False), gr.update(visible=True)
        
        history_a.append((chat_b, chat_a))

        prompt = get_prompt_from_history(history_b, chat_a, system=system_b)
        chat_b = ""
        for text in generate(prompt):
            if stop_generate:
                break

            chat_b += text
            yield history_b + [(chat_a, chat_b)], gr.update(visible=False), gr.update(visible=True)

        
        history_b.append((chat_a, chat_b))

    yield history_b, gr.update(visible=True), gr.update(visible=False)

def undo_history(history):
    return history[:-1]

def swap(name_a, name_b, system_a, system_b, icon_a, icon_b):
    return name_b, name_a, system_b, system_a, icon_b, icon_a

def apply_icon(user_icon, chatbot_icon):
    return gr.update(avatar_images=[user_icon, chatbot_icon])

def simulate(user_avatar=None, chatbot_avatar=None):
    with gr.Blocks() as simulate_interface:
        gr.Markdown("AI同士に会話させるタブです。")
        chatbot = gr.Chatbot(layout="panel")

        with gr.Accordion("System setting"):
            with gr.Row():
                with gr.Group():
                    name_a = gr.Textbox(label="Name A", placeholder="Enter your name here...", lines=1)
                    icon_a = gr.Image(label="Icon A", type="filepath")
                    system_a = gr.Textbox(label="System A", placeholder="Enter your message here...", lines=3)
                with gr.Group():
                    name_b = gr.Textbox(label="Name B", placeholder="Enter your name here...", lines=1)
                    icon_b = gr.Image(label="Icon B", type="filepath")
                    system_b = gr.Textbox(label="System B", placeholder="Enter your message here...", lines=3)
            icon_apply_button = gr.Button("Apply Icon", variant="primary")
            first_message = gr.Textbox(label="First message", placeholder="Enter your message here...", lines=2)
            swap_button = gr.Button("Swap")

        n_completion = gr.Slider(1, 20, value=1, step=1, label="Number of completion")

        generate_button = gr.Button("Generate", variant="primary")
        stop_button = gr.Button("Stop", visible=False)
    
        with gr.Row():    
            undo_button_chat = gr.Button("Undo")
            clear_button = gr.ClearButton([chatbot])

        view_button = gr.Button("View", variant="secondary")
        view_text = gr.Markdown("")

        generate_button.click(chat_handler, inputs=[chatbot, system_a, system_b, first_message, n_completion], outputs=[chatbot, generate_button, stop_button])
        stop_button.click(stop)

        swap_button.click(swap, inputs=[name_a, name_b, system_a, system_b, icon_a, icon_b], outputs=[name_a, name_b, system_a, system_b, icon_a, icon_b])
        undo_button_chat.click(undo_history, inputs=[chatbot], outputs=[chatbot])

        view_button.click(view, inputs=[chatbot, name_a, name_b, icon_a, icon_b, system_a, system_b], outputs=[view_text])

        icon_apply_button.click(apply_icon, inputs=[icon_a, icon_b], outputs=[chatbot])

    return simulate_interface