import gradio as gr
from tabs.setting import setting
from tabs.completion import completion
from tabs.chat import chat
from tabs.simulate import simulate
from tabs.likelihood import likelihood
from tabs.typo_checker import typo_checker
from tabs.arena import arena
from tabs.rag import rag
from tabs.problem import problem
import os

if __name__ == "__main__":
    import argparse

    
    os.makedirs("tmp", exist_ok=True)
    
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
    likelihood_interface = likelihood()
    typo_checker_interface = typo_checker()
    arena_interface = arena()
    rag_interface = rag()
    problem_interface = problem()

    demo = gr.TabbedInterface(
        [
            chat_interface, 
            completion_interface, 
            simulate_interface, 
            likelihood_interface, 
            typo_checker_interface, 
            arena_interface,
            rag_interface,
            problem_interface,
            setting_interface,
        ], 
        [
            "Chat", 
            "Completion",
            "Simulate", 
            "Likelihood",
            "Typo Checker",
            "Arena",
            "RAG",
            "Problem",
            "Setting",
        ],
        theme=gr.themes.Base()
    )

    demo.launch(share = args.share, allowed_paths=["/"])