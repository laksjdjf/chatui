from llama_cpp import Llama
import gradio as gr
import os
from dataclasses import dataclass
import torch
import numpy as np
from tabs.templates import get_template, template_list

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
    debug: bool = False

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
        debug: bool,
    ):
        self.system = system
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.repeat_penalty = repeat_penalty
        self.system_template = system_template
        self.user_template = user_template
        self.chatbot_template = chatbot_template 
        self.debug = debug

        return self.__repr__()
    
    @property
    def system_prefix(self):
        return self.system_template.split("{system}")[0]
    
    @property
    def system_suffix(self):
        return self.system_template.split("{system}")[1]
    
    @property
    def user_prefix(self):
        return self.user_template.split("{user}")[0]
    
    @property
    def user_suffix(self):
        return self.user_template.split("{user}")[1]
    
    @property
    def chatbot_prefix(self):
        return self.chatbot_template.split("{chatbot}")[0]
    
    @property
    def chatbot_suffix(self):
        return self.chatbot_template.split("{chatbot}")[1]

config = ChatConfig()

def generate(prompt:str):
    global model, config
    if config.debug:
        print(f"Prompt: {prompt}")

    for chunk in model(
        prompt,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        repeat_penalty=config.repeat_penalty,
        stream=True,
    ):
        yield chunk["choices"][0]["text"]

def find_first_difference_index(list1, list2):
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] != list2[i]:
            return i
    
    return min_length

def eval_output(input: str, outputs: list):
    global model, config
    input_tokens = model.tokenize(input.encode("utf-8"), special=True) # bos含む

    # 入力トークンが変更された位置を計算
    first_diff_index = find_first_difference_index(model._input_ids.tolist(), input_tokens)
    
    if first_diff_index < len(input_tokens):
        model.n_tokens = first_diff_index # 変更されていないトークンまでは維持
        model.eval(input_tokens[first_diff_index:]) # 変更されたトークンを評価

    log_likelihoods, likelihoods, num_tokens = [], [], []

    for output in outputs:
        output_tokens = model.tokenize(output.encode("utf-8"), add_bos=False, special=True)
        model.eval(output_tokens)
        
        logprobs = Llama.logits_to_logprobs(model.eval_logits) # 対数確率

        # inputの最後（outputの最初の予測）からoutputの最後から二番目（outputの最後の予測）までの対数確率を取得
        logprobs_target = logprobs[range(len(input_tokens)-1, len(input_tokens) + len(output_tokens) - 1), torch.tensor(output_tokens)]
        
        if config.debug:
            print(logprobs_target.shape)

        log_likelihood = logprobs_target.sum()
        likelihood = np.exp(log_likelihood)

        log_likelihoods.append(log_likelihood.item())
        likelihoods.append(likelihood.item())
        num_tokens.append(len(output_tokens))

        model.n_tokens = len(input_tokens) # inputまで評価した状態に戻す。

    return likelihoods, log_likelihoods, num_tokens

def eval_text(text:str, threshold:float):
    global model, config
    tokens = model.tokenize(text.encode("utf-8"), special=True)

    # 入力トークンが変更された位置を計算
    first_diff_index = find_first_difference_index(model._input_ids.tolist(), tokens)
    
    if first_diff_index < len(tokens):
        model.n_tokens = first_diff_index # 変更されていないトークンまでは維持
        model.eval(tokens[first_diff_index:]) # 変更されたトークンを評価
    
    logprobs = Llama.logits_to_logprobs(model.eval_logits)
    
    # 各トークンの対数確率
    logprobs_target = logprobs[range(0, len(tokens) - 1), torch.tensor(tokens[1:])]
    
    if config.debug:
        print(logprobs_target.shape)

    logprobs_max = logprobs.max(axis=1)[:1]

    # 確率が最大値に比べて閾値以下のトークンをtypoとして返す
    # exp(logprobs_target) < exp(logprobs_max) * thresholdの対数をとっています。
    typo = logprobs_target < (logprobs_max + np.log(threshold))
    typo = typo.tolist()
    
    log_likelihood = logprobs_target.sum()
    perplexity = np.exp(-log_likelihood / (len(tokens) - 1)) # 確率の逆数の相乗平均
    
    text_splited = []
    # detokenize
    for x in tokens[1:]:
        try:
            text_splited.append(model.detokenize([x]).decode("utf-8"))
        except:
            text_splited.append("◆")

    return perplexity, typo, text_splited

def get_prompt(user, post_prompt = None):
    '''
    system + user + chatbot_prefix
    or
    post_prompt + user + chatbot_prefix
    '''
    if post_prompt:
        prompt = post_prompt
    else:
        prompt = config.system_template.format(system=config.system)

    prompt += config.user_template.format(user=user)
    prompt += config.chatbot_prefix
    return prompt

def get_prompt_from_history(history, user=None, chatbot_beginning="", generate_input=False, system=None):
    global config
    '''
    if user is not None and not generate_input:
        system + user_1 + chatbot_1 + ... + user_n + chatbot_n + user + chatbot_prefix + chatbot_beginning
    elif user is None and not generate input: （chatbotに続きを書かせる）
        system + user_1 + chatbot_1 + ... + user_n + chatbot_prefix + chatbot_n
    elif generate_input:
        system + user_1 + chatbot_1 + ... + user_n + chatbot_n + user_prefix + user
        
    '''
    prompt = config.system_template.format(system=system if system else config.system)
    for i, (us, cb) in enumerate(history):
        prompt += config.user_template.format(user=us)
        if user is None and (i == len(history) - 1):
            prompt += config.chatbot_prefix + cb
        else:
            prompt += config.chatbot_template.format(chatbot=cb)

    if user and not generate_input:
        prompt += config.user_template.format(user=user)
        prompt += config.chatbot_prefix + chatbot_beginning

    if generate_input:
        prompt += config.user_prefix + user

    return prompt

def setting(model_dir):
    global model, config

    def load_model(model_name, ngl, ctx, ts, n_batch, flash_attn):
        global model
        model_path = os.path.join(model_dir, model_name)
        ts = [float(x) for x in ts.split(",")] if ts else None
        model = Llama(
            model_path=model_path,
            n_gpu_layers=ngl,
            n_batch=n_batch,
            tensor_split=ts,
            n_ctx=ctx,
            flash_attn=flash_attn,
            logits_all = True,
        )

        return "Model loaded successfully."
        
    with gr.Blocks() as setting_interfate:
        with gr.Accordion("モデルのロード"):
            model_name = gr.Dropdown([model for model in os.listdir(model_dir)], label="model_name")
            ngl = gr.Slider(label="n_gpu_layers", minimum=0, maximum=256, step=1, value=256)
            ctx = gr.Slider(label="n_ctx", minimum=256, maximum=65536, step=256, value = 4096)
            ts = gr.Textbox(label="tensor_split")
            n_batch = gr.Slider(label="n_batch", minimum=32, maximum=4096, step=32, value=512)
            flash_attn = gr.Checkbox(label="flash_attn", value=False)
            output = gr.Textbox(label="output", value="")

            with gr.Row():
                load_button = gr.Button(value="Load", variant="primary")
                clear_button = gr.Button(value="Clear", variant="secondary")

        load_button.click(
            load_model,
            inputs=[model_name, ngl, ctx, ts, n_batch, flash_attn],
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
            debug = gr.Checkbox(label="debug", value=False)

            output = gr.Textbox(label="output", interactive=False)
            setting_list = [system, temperature, top_p, max_tokens, repeat_penalty, system_template, user_template, chatbot_template, debug]

            template_dropdown = gr.Dropdown(template_list, label="template_list")

        for setting in setting_list:
            setting.change(config, inputs=setting_list, outputs=output)
        template_dropdown.change(get_template, inputs=template_dropdown, outputs=[system_template, user_template, chatbot_template])    
        
    return setting_interfate

if __name__ == "__main__":
    import sys
    setting_interfate = setting(sys.argv[1])
    setting_interfate.launch()