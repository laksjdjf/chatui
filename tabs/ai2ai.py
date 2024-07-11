import gradio as gr
from modules.llama_process import generate, get_prompt_from_messages, load_state, save_state, get_n_tokens
from modules.utils import view

stop_generate = False

def stop():
    global stop_generate
    stop_generate = True

def ai2ai_handler(history, system_a, system_b, first_message, n_completion, cache_state_threshold):
    global stop_generate
    stop_generate = False
    state = None

    for i in range(n_completion):
        
        # Bが話しかけてAが返事をする。
        messages = [{"role": "system", "content": system_a}, {"role": "user", "content": first_message}]
        for message_a, message_b in history:
            messages.append({"role": "assistant", "content": message_a})
            messages.append({"role": "user", "content": message_b})
        input_message = {"role": "assistant", "content": ""}
        prompt = get_prompt_from_messages(messages, input_message, add_system=False)
        
        if stop_generate:
            break

        current_message_a = ""

        if cache_state_threshold < get_n_tokens():
            state = save_state()
        else:
            state = None
        
        if state is not None:
            load_state(state)

        for text in generate(prompt):
            if stop_generate:
                break

            current_message_a += text
            yield history + [(current_message_a, "")], gr.update(visible=False), gr.update(visible=True)

        if cache_state_threshold < get_n_tokens():
            state = save_state()

        # Aが話しかけてBが返事をする。
        messages = [{"role": "system", "content": system_b}]
        for message_a, message_b in history:
            messages.append({"role": "user", "content": message_a})
            messages.append({"role": "assistant", "content": message_b})
        messages.append({"role": "user", "content": current_message_a})
        input_message = {"role": "assistant", "content": ""}
        prompt = get_prompt_from_messages(messages, input_message, add_system=False)

        current_message_b = ""

        if state is not None:
            load_state(state)

        for text in generate(prompt):
            if stop_generate:
                break

            current_message_b += text
            yield history + [(current_message_a, current_message_b)], gr.update(visible=False), gr.update(visible=True)
        
        if cache_state_threshold < get_n_tokens():
            state = save_state()

        history.append((current_message_a, current_message_b))

    yield history, gr.update(visible=True), gr.update(visible=False)

def undo_history(history):
    return history[:-1]

def swap(name_a, name_b, system_a, system_b, icon_a, icon_b):
    return name_b, name_a, system_b, system_a, icon_b, icon_a

def apply_icon(user_icon, chatbot_icon):
    return gr.update(avatar_images=[user_icon, chatbot_icon])

def ai2ai():
    with gr.Blocks() as ai2ai_interface:
        gr.Markdown("AI同士に会話させるタブです。")
        chatbot = gr.Chatbot(height=600, layout="panel")

        generate_button = gr.Button("Generate", variant="primary")
        stop_button = gr.Button("Stop", visible=False)

        with gr.Accordion("System setting"):
            with gr.Row():                
                with gr.Group():
                    name_b = gr.Textbox(label="Name B", placeholder="Enter your name here...", lines=1)
                    icon_b = gr.Image(label="Icon B", type="filepath")
                    system_b = gr.Textbox(label="System B", placeholder="Enter your message here...", lines=3)
                with gr.Group():
                    name_a = gr.Textbox(label="Name A", placeholder="Enter your name here...", lines=1)
                    icon_a = gr.Image(label="Icon A", type="filepath")
                    system_a = gr.Textbox(label="System A", placeholder="Enter your message here...", lines=3)
            icon_apply_button = gr.Button("Apply Icon", variant="primary")
            first_message = gr.Textbox(label="First message", placeholder="Enter your message here...", lines=2)
            swap_button = gr.Button("Swap")

            n_completion = gr.Slider(1, 20, value=1, step=1, label="Number of completion")
            cache_state_threshold = gr.Slider(0, 65536, value=65536, step=1, label="Cache state if num_tokens >=")
    
        with gr.Row():    
            undo_button_chat = gr.Button("Undo")
            clear_button = gr.ClearButton([chatbot])

        view_button = gr.Button("View", variant="secondary")
        view_text = gr.Markdown("")

        generate_button.click(ai2ai_handler, inputs=[chatbot, system_a, system_b, first_message, n_completion, cache_state_threshold], outputs=[chatbot, generate_button, stop_button])
        stop_button.click(stop)

        swap_button.click(swap, inputs=[name_a, name_b, system_a, system_b, icon_a, icon_b], outputs=[name_a, name_b, system_a, system_b, icon_a, icon_b])
        undo_button_chat.click(undo_history, inputs=[chatbot], outputs=[chatbot])

        view_button.click(view, inputs=[chatbot, name_a, name_b, icon_a, icon_b, system_a, system_b], outputs=[view_text])

        icon_apply_button.click(apply_icon, inputs=[icon_a, icon_b], outputs=[chatbot])

    return ai2ai_interface