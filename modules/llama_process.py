from llama_cpp import Llama, LogitsProcessor
from llama_cpp.llama_chat_format import Llava15ChatHandler, Llava16ChatHandler
from llama_cpp.llama_grammar import LlamaGrammar, JSON_GBNF, LIST_GBNF
from llama_cpp.llama_cpp import GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F16
from modules.config import ChatConfig
import os
import torch
import numpy as np
import re

model = None
config = ChatConfig()
logits_processor = None

GGML_TYPE = {
    "q4_0": GGML_TYPE_Q4_0,
    "q8_0": GGML_TYPE_Q8_0,
    "f16": GGML_TYPE_F16
}

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

PROBLEM_DICT_GBNF = r"""
root   ::= object

object ::=
  "{" ws "\"reason\"" ws ":" ws string "," ws "\"answer\"" ws ":" ws answer "}" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

answer ::= "\"" ("A" | "B" | "C" | "D" | "E") "\"" ws

ws ::= ([ \t\n] ws)?
"""

ELYZA_EVAL_GBNF = r"""
root   ::= object

object ::=
  "{" "\"reason\"" ":" ws string "," "\"grade\"" ":" grade "}"
string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

grade ::= "\"" ("1" | "2" | "3" | "4" | "5") "\"" ws

ws ::= ([ \t\n] ws)?
"""

# Chat GPTに作らせたので、よくわからない。
PATTERNS = {
    'ひらがな': '[\u3040-\u309F]',
    'カタカナ': '[\u30A0-\u30FF]',
    '漢字': '[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u20000-\u2A6DF\u2A700-\u2B73F\u2B740-\u2B81F\u2B820-\u2CEAF\u2CEB0-\u2EBEF]',
    'アルファベット': '[a-zA-Z\u00C0-\u00FF\u0100-\u017F\u0180-\u024F]',
    'ハングル': '[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uD7B0-\uD7FF]',
    'アラビア文字': '[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]',
    'キリル文字': '[\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F]',
    'ギリシャ文字': '[\u0370-\u03FF\u1F00-\u1FFF]',
    'デーヴァナーガリー文字': '[\u0900-\u097F]',
    'タイ文字': '[\u0E00-\u0E7F]',
    'ヘブライ文字': '[\u0590-\u05FF]',
    'エチオピア文字': '[\u1200-\u137F]',
    '絵文字': '[\u1F600-\u1F64F\u1F300-\u1F5FF\u1F680-\u1F6FF\u1F900-\u1F9FF\u2600-\u26FF]'
}


def get_gbnf(template):
    if template == "japanese":
        return JAPANESE_SIMPLE_GBNF
    elif template == "list":
        return LIST_GBNF
    elif template == "json":
        return JSON_GBNF
    elif template == "problem_dict":
        return PROBLEM_DICT_GBNF
    elif template == "elyza_eval":
        return ELYZA_EVAL_GBNF
    else:
        return ""
    
def load_config(*args):
    global config
    config = ChatConfig(*args)
    return config.__repr__()

def get_config():
    return config

class BanLogitsProcessor(LogitsProcessor):
    def __init__(self, ban_ids):
        self.ban_ids = ban_ids

    def __call__(self, input_ids, scores):
        scores[self.ban_ids] = -np.inf
        return scores
    
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

def load_logits_processor(languages):
    global logits_processor
    if languages:
        ban_ids = get_ban_token_ids(languages)
        if len(ban_ids) == 0:
            logits_processor = None
            return "No ban tokens found."
        else:
            logits_processor = BanLogitsProcessor(ban_ids)
            return f"Number of ban tokens: {len(ban_ids)}"
    logits_processor = None
    return "No languages selected."

def load_llava_handler(llava_handler, mmproj_path):
    if llava_handler == "Llava16ChatHandler":
        return Llava16ChatHandler(clip_model_path=mmproj_path)
    elif llava_handler == "Llava15ChatHandler":
        return Llava15ChatHandler(clip_model_path=mmproj_path)
    return None
    
def load_model(model_dir, model_name, mmproj_name, llava_handler, lora_name, ngl, ctx, ts, n_batch, flash_attn, no_kv_offload, type_kv, logits_all):
    global model
    del model
    model = None # error handling
    model_path = os.path.join(model_dir, model_name)
    mmproj_path = os.path.join(model_dir, "mmprojs", mmproj_name) if mmproj_name != "None" else None
    llava_handler = load_llava_handler(llava_handler, mmproj_path)
    lora_path = os.path.join(model_dir, "loras", lora_name) if lora_name != "None" else None

    ts = [float(x) for x in ts.split(",")] if ts else None
    model = Llama(
        model_path=model_path,
        chat_handler=llava_handler,
        lora_path=lora_path,
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

def clear_model():
    global model
    del model
    model = None
    return "Model cleared."

def load_state(state):
    global model
    model.load_state(state)

def save_state():
    global model
    return model.save_state()

def get_n_tokens(): 
    global model
    return model.n_tokens

def get_prompt_from_messages(messages, input_message, add_system=False):
    return config.get_prompt_from_messages(messages, input_message, add_system=add_system)

def get_default_prompt(user, add_system=True):
    messages = [{"role": "user", "content": user}]
    input_message = {"role": "assistant", "content": ""}
    prompt = get_prompt_from_messages(messages, input_message, add_system=add_system)
    return prompt

def generate(prompt):
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
        min_p=config.min_p,
        top_k=config.top_k,
        repeat_penalty=config.repeat_penalty,
        grammar=grammar,
        logits_processor=logits_processor,
        stream=True,
    ):
        yield chunk["choices"][0]["text"]

def generate_chat(messages):
    global model, config, logits_processor
    if config.debug:
        print(f"Prompt: {messages}")

    if config.grammar:
        grammar = LlamaGrammar.from_string(config.grammar)
    else:
        grammar = None

    for chunk in model.create_chat_completion(
        messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        min_p=config.min_p,
        top_k=config.top_k,
        repeat_penalty=config.repeat_penalty,
        grammar=grammar,
        logits_processor=logits_processor,
        stream=True,
    ):
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            yield delta["content"]

def find_first_difference_index(list1, list2):
    min_length = min(len(list1), len(list2))
    for i in range(min_length):
        if list1[i] != list2[i]:
            return i
    return min_length

def eval_output(input: str, outputs: list = []):
    global model, config

    if not model.context_params.logits_all:
        raise ValueError("Please set logits_all to True in the model configuration.")
        
    input_tokens = model.tokenize(input.encode("utf-8"), special=True) # bos含む

    # 入力トークンが変更された位置を計算
    first_diff_index = find_first_difference_index(model._input_ids.tolist(), input_tokens)
    
    if first_diff_index < len(input_tokens):
        model.n_tokens = first_diff_index # 変更されていないトークンまでは維持
        model.eval(input_tokens[first_diff_index:]) # 変更されたトークンを評価

    log_likelihoods, likelihoods, num_tokens = [], [], []

    for output in outputs:
        output_tokens = model.tokenize(output.encode("utf-8"), add_bos=False, special=True)
        if len(output_tokens) > 1:
            model.eval(output_tokens[:-1]) # last token is not needed for evaluation
        
        scores = model._scores[range(len(input_tokens)-1, len(input_tokens) + len(output_tokens) - 1)]
        logprobs = torch.nn.functional.log_softmax(torch.from_numpy(scores), dim=-1).numpy()

        # inputの最後（outputの最初の予測）からoutputの最後から二番目（outputの最後の予測）までの対数確率を取得
        logprobs_target = logprobs[:, output_tokens]

        log_likelihood = logprobs_target.sum()
        likelihood = np.exp(log_likelihood)

        log_likelihoods.append(log_likelihood.item())
        likelihoods.append(likelihood.item())
        num_tokens.append(len(output_tokens))

        model.n_tokens = len(input_tokens) # inputまで評価した状態に戻す。

    return likelihoods, log_likelihoods, num_tokens

def eval_text(text:str, relative: bool = False):
    global model, config

    if not model.context_params.logits_all:
        raise ValueError("Please set logits_all to True in the model configuration.")

    tokens = model.tokenize(text.encode("utf-8"), special=True)
    first_diff_index = find_first_difference_index(model._input_ids.tolist(), tokens)
    
    if first_diff_index < len(tokens):
        model.n_tokens = first_diff_index # 変更されていないトークンまでは維持
        model.eval(tokens[first_diff_index:]) # 変更されたトークンを評価
    
    scores = model._scores[:-1]
    logprobs = torch.nn.functional.log_softmax(torch.from_numpy(scores), dim=-1).numpy()
    
    # 各トークンの対数確率
    logprobs_target = logprobs[range(0, len(tokens) - 1), tokens[1:]]
    logprobs_max = logprobs.max(axis=1)[:1]

    if relative:
        result = np.exp(logprobs_target - logprobs_max).tolist()
    else:
        result = np.exp(logprobs_target).tolist()
    
    log_likelihood = logprobs_target.sum()
    perplexity = np.exp(-log_likelihood / (len(tokens) - 1)) # 確率の逆数の相乗平均
    
    text_splited = []
    # detokenize
    for i in range(1, len(tokens)):
        try:
            text_splited.append(model.detokenize([tokens[i]], prev_tokens=tokens[:i]).decode("utf-8"))
        except:
            text_splited.append("◆")

    return perplexity, result, text_splited

def eval_choice(input: str, choices: list):
    global model, config

    input_tokens = model.tokenize(input.encode("utf-8"), special=True) # bos含む

    # 入力トークンが変更された位置を計算
    first_diff_index = find_first_difference_index(model._input_ids.tolist(), input_tokens)
    
    if first_diff_index < len(input_tokens):
        model.n_tokens = first_diff_index # 変更されていないトークンまでは維持
        model.eval(input_tokens[first_diff_index:]) # 変更されたトークンを評価
    
    output_tokens = [model.tokenize(choice.encode("utf-8"), add_bos=False, special=True)[0] for choice in choices]

    choice = model._scores[-1][output_tokens].argmax()
    return choices[choice]

def get_token_split(text: str):
    global model
    tokens = model.tokenize(text.encode("utf-8"), special=True)
    text_splited = ""
    for i in range(1, len(tokens)):
        try:
            word = model.detokenize([tokens[i]], prev_tokens=tokens[:i]).decode("utf-8")
        except:
            word = "◆"
        color = ["plum", "lightgreen", "yellow", "pink", "lightblue"][i % 5]
        if word == "\n":
            text_splited += "\n\n"
        else:
            text_splited += f"<span style='background-color:{color}'>{word}</span> "

    return text_splited, str(tokens), f"# Tokens {len(tokens)}   Characters {len(text)}"