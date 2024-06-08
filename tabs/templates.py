cohere = (
    "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system}<|END_OF_TURN_TOKEN|>",
    "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user}<|END_OF_TURN_TOKEN|>",
    "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{chatbot}<|END_OF_TURN_TOKEN|>",
)

phi3 = (
    "{system}",
    "<|user|>\n{user}<|end|>\n",
    "<|assistant|>\n{chatbot}<|end|>\n",
)

swallow = (
    "### 指示:\n{system}\n\n",
    "### 入力:\n{user}\n\n",
    "### 応答:\n{chatbot}\n\n",
)

llama2 = (
    "<<SYS>>\n{system}\n<</SYS>>\n\n", # 本来は[INST]が最初にくるが・・・
    "[INST]{user}[/INST]",
    "{chatbot}",
)

llama3 = (
    "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
    "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>",
    "<|start_header_id|>chatbot<|end_header_id|>\n\n{chatbot}<|eot_id|>",

)

qwen = (
    "<|im_start|>system\n{system}<|im_end|>",
    "<|im_start|>user\n{user}<|im_end|>",
    "<|im_start|>assistant\n{chatbot}<|im_end|>",
)

template_list = ["cohere", "phi3", "swallow", "llama2", "llama3", "qwen"]

def get_template(template):
    if template == "cohere":
        return cohere
    elif template == "phi3":
        return phi3
    elif template == "swallow":
        return swallow
    elif template == "llama2":
        return llama2
    elif template == "llama3":
        return llama3
    elif template == "qwen":
        return qwen
    return 
