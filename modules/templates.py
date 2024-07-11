cohere = (
    "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
    "<|END_OF_TURN_TOKEN|>",
    "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
    "<|END_OF_TURN_TOKEN|>",
    "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
    "<|END_OF_TURN_TOKEN|>",
)

phi3 = (
    "",
    "",
    "<|user|>\n",
    "<|end|>\n",
    "<|assistant|>\n",
    "<|end|>\n",
)

swallow = (
    "",
    "",
    "### 指示:\n",
    "\n\n",
    "### 応答:\n",
    "\n\n",
)

llama2 = (
    "<<SYS>>\n",
    "\n<</SYS>>\n\n",
    "[INST]",
    "[/INST]",
    "",
    "",
)

llama3 = (
    "<|start_header_id|>system<|end_header_id|>\n\n",
    "<|eot_id|>",
    "<|start_header_id|>user<|end_header_id|>\n\n",
    "<|eot_id|>",
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "<|eot_id|>",
)

gemma = (
    "",
    "",
    "<start_of_turn>user\n",
    "<end_of_turn>\n",
    "<start_of_turn>model\n",
    "<end_of_turn>\n",
)

qwen = (
    "<|im_start|>system\n",
    "<|im_end|>\n",
    "<|im_start|>user\n",
    "<|im_end|>\n",
    "<|im_start|>assistant\n",
    "<|im_end|>\n",
)

template_list = ["cohere", "phi3", "swallow", "llama2", "llama3", "gemma", "qwen"]

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
    elif template == "gemma":
        return gemma
    elif template == "qwen":
        return qwen
    return 