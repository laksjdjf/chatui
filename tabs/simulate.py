import gradio as gr
from tabs.setting import generate, get_prompt_for_simulation

stop_generate = False

def stop():
    global stop_generate
    stop_generate = True

def chat_handler(history, system_a, system_b, first_message, n_completion):
    global stop_generate
    stop_generate = False

    if len(history) > 0:
        first_message = history[-1][1]

    for i in range(n_completion):
        chat_a, chat_b = "", ""
        prompt = get_prompt_for_simulation(history, system_a, first_message)
        
        if stop_generate:
            break

        for text in generate(prompt):
            if stop_generate:
                break

            chat_a += text
            yield history + [(chat_a, "")], gr.update(visible=False), gr.update(visible=True)
        
        prompt = get_prompt_for_simulation(history, system_b, chat_a, swap=True)

        for text in generate(prompt):
            if stop_generate:
                break

            chat_b += text
            yield history + [(chat_a, chat_b)], gr.update(visible=True), gr.update(visible=False)

        history.append((chat_a, chat_b))

    yield history, gr.update(visible=True), gr.update(visible=False)

def undo_history(history):
    return history[:-1]

def view(history, name_a, name_b):
    return "\n\n".join([f'<div style="color: navy">{name_a}:{user.replace("\n", "")}</div>\n\n<div style="color: maroon">{name_b}:{chatbot.replace("\n", "")}</div>' for user, chatbot in history])

def swap(name_a, name_b, system_a, system_b):
    return name_b, name_a, system_b, system_a

def simulate(user_avatar=None, chatbot_avatar=None):
    with gr.Blocks() as simulate_interface:
        gr.Markdown("AI同士に会話させるタブです。")
        chatbot = gr.Chatbot(layout="panel")

        with gr.Accordion("System setting"):
            with gr.Row():
                with gr.Group():
                    name_a = gr.Textbox(label="Name A", placeholder="Enter your name here...", lines=1)
                    system_a = gr.Textbox(label="System A", placeholder="Enter your message here...", lines=3)
                with gr.Group():
                    name_b = gr.Textbox(label="Name B", placeholder="Enter your name here...", lines=1)
                    system_b = gr.Textbox(label="System A", placeholder="Enter your message here...", lines=3)
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

        swap_button.click(swap, inputs=[name_a, name_b, system_a, system_b], outputs=[name_a, name_b, system_a, system_b])
        undo_button_chat.click(undo_history, inputs=[chatbot], outputs=[chatbot])

        view_button.click(view, inputs=[chatbot, name_a, name_b], outputs=[view_text])

    return simulate_interface