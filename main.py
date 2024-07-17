import gradio as gr
from tabs.setting import setting
from tabs.chat import chat
from tabs.playground import playground
from tabs.completion import completion
from tabs.likelihood import likelihood
from tabs.eval_sentence import eval_sentence
from tabs.problem import problem
from tabs.arena import arena
from tabs.rag import rag
from tabs.ai2ai import ai2ai
from tabs.tokenizer import tokenizer
from tabs.multimodal import multimodal
from tabs.questions import questions
import os

if __name__ == "__main__":
    import argparse

    os.makedirs("tmp", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--model_dir", "-m", help="path to model directory")
    args = parser.parse_args()

    demo = gr.TabbedInterface(
        [
            chat(),
            playground(),
            completion(),
            multimodal(),
            likelihood(),
            eval_sentence(),
            problem(),
            questions(),
            arena(),
            rag(),
            ai2ai(),
            tokenizer(),
            setting(args.model_dir),
        ], 
        [
            "Chat",
            "Playground",
            "Completion",
            "Multimodal",
            "Likelihood",
            "EvalSentence",
            "Problem",
            "Questions",
            "Arena",
            "RAG",
            "AI2AI",
            "Tokenizer",
            "Setting",
        ],
        theme=gr.themes.Base()
    )

    demo.launch(share = args.share, allowed_paths=["/"])