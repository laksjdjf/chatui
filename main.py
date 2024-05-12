import gradio as gr
from tabs.setting import setting
from tabs.completion import completion
from tabs.chat import chat
from tabs.simulate import simulate

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--model_dir", "-m", help="path to model directory")
    parser.add_argument("--user_avatar", "-ua", help="path to user image")
    parser.add_argument("--chatbot_avatar", "-ca", help="path to chatbot image")
    args = parser.parse_args()

    chat_interface = chat(args.user_avatar, args.chatbot_avatar)
    completion_interface = completion()
    setting_interface = setting(args.model_dir)
    simulate_interface = simulate()
    

    demo = gr.TabbedInterface([chat_interface, completion_interface, simulate_interface, setting_interface], ["Chat", "Completion", "Simulate", "Setting"], theme=gr.themes.Base())

    demo.launch(share = args.share)