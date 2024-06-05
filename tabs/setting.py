from llama_cpp import Llama, LogitsProcessor
from llama_cpp.llama_grammar import LlamaGrammar, JSON_GBNF, LIST_GBNF
from llama_cpp.llama_cpp import GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F16
import gradio as gr
import os
from dataclasses import dataclass
import torch
import numpy as np
import re
from tabs.templates import get_template, template_list

model = None

# https://github.com/ggerganov/llama.cpp/blob/master/grammars/japanese.gbnf
JAPANESE_SIMPLE_GBNF = r"""
root        ::= (jp-char | space)+
jp-char     ::= hiragana | katakana | punctuation | cjk
hiragana    ::= [ぁ-ゟ]
katakana    ::= [ァ-ヿ]
punctuation ::= [、-〽]
cjk         ::= [一-龯]
space       ::= [ \t\n]
"""

GGML_TYPE = {
    "q4_0": GGML_TYPE_Q4_0,
    "q8_0": GGML_TYPE_Q8_0,
    "f16": GGML_TYPE_F16
}

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
    grammar: str = ""
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
        grammar: str,
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
        self.grammar = grammar
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

class BanLogitsProcessor(LogitsProcessor):
    def __init__(self, ban_ids):
        self.ban_ids = ban_ids

    def __call__(self, input_ids, scores):
        scores[self.ban_ids] = -np.inf
        return scores

logits_processor = None

PATTERNS = {
    'ひらがな': '[\u3040-\u309F]',
    'カタカナ': '[\u30A0-\u30FF]',
    '漢字': '[\u4E00-\u9FFF]',
    'アルファベット': '[a-zA-Z]',
    'ハングル': '[\uAC00-\uD7AF]',
    'アラビア文字': '[\u0600-\u06FF]',
    'キリル文字': '[\u0400-\u04FF]',
    'ギリシャ文字': '[\u0370-\u03FF]',
    'デーヴァナーガリー文字': '[\u0900-\u097F]',
    'タイ文字': '[\u0E00-\u0E7F]',
    'ヘブライ文字': '[\u0590-\u05FF]',
    'エチオピア文字': '[\u1200-\u137F]'   
}

def contains_character_set(text, languages):
    for language in languages:
        compiled_pattern = re.compile(PATTERNS[language])
        if bool(compiled_pattern.search(text)):
            return True
    return False

def get_ban_token_ids(languages):
    global model
    ban_tokens = []
    for i in range(model.n_vocab()):
        try:
            text = model.detokenize([i]).decode("utf-8")
        except:
            continue
        if contains_character_set(text, languages):
            ban_tokens.append(i)
    return ban_tokens

def set_logits_processor(languages):
    global logits_processor
    if languages:
        ban_ids = get_ban_token_ids(languages)
        logits_processor = BanLogitsProcessor(ban_ids)
    else:
        logits_processor = None
    
    return f"Number of ban tokens: {len(ban_ids)}"

def generate(prompt:str):
    global model, config, logits_processor
    if config.debug:
        print(f"Prompt: {prompt}")

    if config.grammar:
        grammar = LlamaGrammar.from_string(config.grammar)
    else:
        grammar = None

    for chunk in model(
        prompt,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        repeat_penalty=config.repeat_penalty,
        grammar=grammar,
        logits_processor=logits_processor,
        stream=True,
    ):
        yield chunk["choices"][0]["text"]

def find_first_difference_index(list1, list2):
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] != list2[i]:
            return i
    
    return min_length

def eval_output(input: str, outputs: list = []):
    global model, config
    
    if not model.context_params.logits_all:
        raise ValueError("logits_all must be True.")
    
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

    if not model.context_params.logits_all:
        raise ValueError("logits_all must be True.")

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

def get_gbnf(template):
    if template == "japanese":
        return JAPANESE_SIMPLE_GBNF
    elif template == "list":
        return LIST_GBNF
    elif template == "json":
        return JSON_GBNF
    else:
        return ""

def view(history, name_a="user", name_b="chatbot", system_a=None, system_b=None):
    if system_a and system_b:
        text = f"{name_a}: {system_a}\n{name_b}: {system_b}\n\n"
    else:
        text = f"system: {config.system}\n\n"
        
    for user, chatbot in history:
        
        # 改行でspanが効かなくなっちゃうので、改行ごとにspanを挟む
        user_split_line = [name_a + ":"] + user.split("\n")
        user = "\n".join([f'<span style="color: navy">{line}</span>' for line in user_split_line])

        chatbot_split_line = [name_b + ":"] + chatbot.split("\n")
        chatbot = "\n".join([f'<span style="color: maroon">{line}</span>' for line in chatbot_split_line])

        text += f'{user}\n\n{chatbot}\n\n'

    return text

def setting(model_dir):
    global model, config

    def load_model(model_name, ngl, ctx, ts, n_batch, flash_attn, no_kv_offload, type_kv, logits_all):
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
            offload_kqv= not no_kv_offload,
            type_k=GGML_TYPE[type_kv],
            type_v=GGML_TYPE[type_kv],
            logits_all=logits_all
        )

        return "Model loaded successfully."
        
    with gr.Blocks() as setting_interfate:
        with gr.Accordion("モデルのロード"):
            model_name = gr.Dropdown([model for model in os.listdir(model_dir)], label="model_name")
            ngl = gr.Slider(label="n_gpu_layers", minimum=0, maximum=256, step=1, value=256)
            ctx = gr.Slider(label="n_ctx", minimum=256, maximum=256000, step=256, value = 4096)
            ts = gr.Textbox(label="tensor_split")
            n_batch = gr.Slider(label="n_batch", minimum=32, maximum=4096, step=32, value=512)
            flash_attn = gr.Checkbox(label="flash_attn", value=False)
            no_kv_offload = gr.Checkbox(label="no_kv_offload", value=False)
            type_kv = gr.Dropdown(["q4_0", "q8_0", "f16"], value="f16", label="type_kv")
            logits_all = gr.Checkbox(label="logits_all", value=True)
            output = gr.Textbox(label="output", value="")

            with gr.Row():
                load_button = gr.Button(value="Load", variant="primary")
                clear_button = gr.Button(value="Clear", variant="secondary")

        load_button.click(
            load_model,
            inputs=[model_name, ngl, ctx, ts, n_batch, flash_attn, no_kv_offload, type_kv, logits_all],
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
            max_tokens = gr.Slider(minimum=1, maximum=256000, step=1, value=256, label="max_tokens")
            repeat_penalty = gr.Slider(minimum=1.0, maximum=2.0, step=0.01, value=1.0, label="repeat_penalty")
            system_template = gr.Textbox(label="system_template", value="<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system}<|END_OF_TURN_TOKEN|>", lines=3)
            user_template = gr.Textbox(label="user_template", value="<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user}<|END_OF_TURN_TOKEN|>", lines=3)
            chatbot_template = gr.Textbox(label="chatbot_template", value="<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{chatbot}<|END_OF_TURN_TOKEN|>", lines=3)
            grammar = gr.Textbox(label="grammar", value="", lines=3)
            debug = gr.Checkbox(label="debug", value=False)

            output = gr.Textbox(label="output", interactive=False)
            setting_list = [system, temperature, top_p, max_tokens, repeat_penalty, system_template, user_template, chatbot_template, grammar, debug]

            template_dropdown = gr.Dropdown(template_list, label="template_list")
            grammar_dropdown = gr.Dropdown(["list", "json", "japanese"], label="grammar_list")

        for setting in setting_list:
            setting.change(config, inputs=setting_list, outputs=output)
        template_dropdown.change(get_template, inputs=template_dropdown, outputs=[system_template, user_template, chatbot_template])    
        grammar_dropdown.change(get_gbnf, inputs=grammar_dropdown, outputs=[grammar])

        with gr.Accordion("Logits Processor"):
            languages_checkbox = gr.CheckboxGroup(list(PATTERNS.keys()), label="check the languages you want to ban")
            logits_processor_load_button = gr.Button("Load", variant="primary")
            logits_processor_output = gr.Textbox(label="number of ban tokens", interactive=False)
        
        logits_processor_load_button.click(
            set_logits_processor,
            inputs=[languages_checkbox],
            outputs=[logits_processor_output]
        )

    return setting_interfate

if __name__ == "__main__":
    import sys
    setting_interfate = setting(sys.argv[1])
    setting_interfate.launch()
