from llama_cpp import Llama
import gradio as gr
import os
from dataclasses import dataclass
import torch
import numpy as np

model = None

@dataclass
class ChatConfig:
    system: str = "あなたは優秀なアシスタントです。"
    temperature: float = 0.8
    top_p: float = 0.9
    max_tokens: int = 256
    repeat_penalty: float = 1.0
    system_template: str = "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system}<|END_OF_TURN_TOKEN|>"
    user_template: str = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user}<|END_OF_TURN_TOKEN|>"
    chatbot_template: str = "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{chatbot}<|END_OF_TURN_TOKEN|>"

    def __call__(
        self,
        system: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        repeat_penalty: float,
        system_template: str,
        user_template: str,
        chatbot_template: str,
    ):
        self.system = system
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.repeat_penalty = repeat_penalty
        self.system_template = system_template
        self.user_template = user_template
        self.chatbot_template = chatbot_template 

        return self.__repr__()

config = ChatConfig()

def generate(prompt):
    global model, config
    for chunk in model(
        prompt,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        repeat_penalty=config.repeat_penalty,
        stream=True,
    ):
        yield chunk["choices"][0]["text"]

def eval_output(input, outputs):
    global model, config
    input_tokens = model.tokenize(input.encode("utf-8"), special=True)

    if model._input_ids.tolist() != input_tokens: # reset and eval if input has changed
        model.reset()
        model.eval(input_tokens)

    log_likelihoods = []
    likelihoods = []
    num_tokens = []

    for output in outputs:
        output_tokens = model.tokenize(output.encode("utf-8"), add_bos=False, special=True)
        model.eval(output_tokens)
        
        logprobs = Llama.logits_to_logprobs(model.eval_logits)
        model.reset()

        # inputの最後から、outputの最後から二番目まで
        logprobs_target = logprobs[range(len(input_tokens)-1, len(input_tokens) + len(output_tokens) - 1), torch.tensor(output_tokens)]
        
        log_likelihood = logprobs_target.sum()
        likelihood = np.exp(log_likelihood)

        log_likelihoods.append(log_likelihood.item())
        likelihoods.append(likelihood.item())
        num_tokens.append(len(output_tokens))

        model.n_tokens = len(input_tokens) # inputまで評価した状態に戻す。

        print(f"{output}:\n  Likelihood: {likelihood:.3f}\n  Log Likelihood: {log_likelihood:.3f}\n num_tokens: {len(output_tokens)})")

    return likelihoods, log_likelihoods, num_tokens

def eval_text(text, threshold):
    global model, config
    tokens = model.tokenize(text.encode("utf-8"), special=True)

    if model._input_ids.tolist() != tokens: # reset and eval if input has changed
        model.reset()
        model.eval(tokens)
    
    logprobs = Llama.logits_to_logprobs(model.eval_logits)

    print(logprobs.shape)
    logprobs_target = logprobs[range(0, len(tokens) - 1), torch.tensor(tokens[1:])]
    logprobs_max = logprobs.max(axis=1)[:1]

    typo = logprobs_target < (logprobs_max + np.log(threshold))
    typo = typo.tolist()
    
    log_likelihood = logprobs_target.sum()
    perplexity = np.exp(-log_likelihood / (len(tokens) - 1))
    
    text_splited = []
    for x in tokens[1:]:
        try:
            text_splited.append(model.detokenize([x]).decode("utf-8"))
        except:
            text_splited.append("◆")

    return perplexity, typo, text_splited

def get_prompt(user, post_prompt = None):
    if post_prompt:
        prompt = post_prompt
    else:
        prompt = config.system_template.format(system=config.system)

    prompt += config.user_template.format(user=user)
    prompt += config.chatbot_template.split("{chatbot}")[0]
    return prompt

def get_prompt_from_history(history, user=None, chatbot_beginning=None):
    prompt = config.system_template.format(system=config.system)
    for i, (us, cb) in enumerate(history):
        prompt += config.user_template.format(user=us)
        if user is None and (i == len(history) - 1):
            prompt += config.chatbot_template.split("{chatbot}")[0] + cb
        else:
            prompt += config.chatbot_template.format(chatbot=cb)

    if user:
        prompt += config.user_template.format(user=user)
        prompt += config.chatbot_template.split("{chatbot}")[0] + chatbot_beginning

    return prompt

def get_prompt_for_simulation(history, system, user, swap=False):
    prompt = config.system_template.format(system=system)
    for i, (us, cb) in enumerate(history):
        if not swap:
            prompt += config.user_template.format(user=us)
            prompt += config.chatbot_template.format(chatbot=cb)
        else:
            prompt += config.chatbot_template.format(chatbot=us)
            prompt += config.user_template.format(user=cb)

    prompt += config.user_template.format(user=user)
    prompt += config.chatbot_template.split("{chatbot}")[0]
    return prompt

def setting(model_dir):
    global model, config
    with gr.Blocks() as setting_interfate:
        with gr.Accordion("モデルのロード"):
            model_name = gr.Dropdown([model for model in os.listdir(model_dir)], label="model_name")
            ngl = gr.Slider(label="n_gpu_layers", minimum=0, maximum=256, step=1, value=256)
            ctx = gr.Slider(label="n_ctx", minimum=256, maximum=65536, step=256, value = 4096)
            ts = gr.Textbox(label="tensor_split")
            n_batch = gr.Slider(label="n_batch", minimum=32, maximum=4096, step=32, value=512)
            output = gr.Textbox(label="output", value="")

            with gr.Row():
                load_button = gr.Button(value="Load", variant="primary")
                clear_button = gr.Button(value="Clear", variant="secondary")

        def load_model(model_name, ngl, ctx, ts, n_batch):
            global model
            model_path = os.path.join(model_dir, model_name)
            ts = [float(x) for x in ts.split(",")] if ts else None
            model = Llama(
                model_path=model_path,
                n_gpu_layers=ngl,
                n_batch=n_batch,
                tensor_split=ts,
                n_ctx=ctx,
                logits_all = True,
            )

            return "Model loaded successfully."

        load_button.click(
            load_model,
            inputs=[model_name, ngl, ctx, ts, n_batch],
            outputs=[output],
        )

        def clear_output():
            global model
            del model
            model = None
            return "Model unloaded."

        clear_button.click(
            clear_output,
            inputs=None,
            outputs=[output],
        )

        with gr.Accordion("生成設定"):
            system = gr.Textbox(label="system", value="あなたは優秀なアシスタントです。", lines=2)
            temperature = gr.Slider(minimum=0, maximum=2, step=0.01, value=0.8, label="temperature")
            top_p = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.9, label="top_p")
            max_tokens = gr.Slider(minimum=1, maximum=65536, step=1, value=256, label="max_tokens")
            repeat_penalty = gr.Slider(minimum=1.0, maximum=2.0, step=0.01, value=1.0, label="repeat_penalty")
            system_template = gr.Textbox(label="system_template", value="<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system}<|END_OF_TURN_TOKEN|>", lines=3)
            user_template = gr.Textbox(label="user_template", value="<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user}<|END_OF_TURN_TOKEN|>", lines=3)
            chatbot_template = gr.Textbox(label="chatbot_template", value="<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{chatbot}<|END_OF_TURN_TOKEN|>", lines=3)

            output = gr.Textbox(label="output", interactive=False)

            setting_list = [system, temperature, top_p, max_tokens, repeat_penalty, system_template, user_template, chatbot_template]

        for setting in setting_list:
            setting.change(config, inputs=setting_list, outputs=output)    
        
    return setting_interfate

if __name__ == "__main__":
    import sys
    setting_interfate = setting(sys.argv[1])
    setting_interfate.launch()