from dataclasses import dataclass

@dataclass
class ChatConfig:
    system: str = "あなたは優秀なアシスタントです。"
    temperature: float = 0.8
    top_p: float = 0.9
    max_tokens: int = 256
    repeat_penalty: float = 1.0
    system_prefix: str = "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>"
    system_suffix:str = "<|END_OF_TURN_TOKEN|>"
    user_prefix: str = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
    user_suffix: str = "<|END_OF_TURN_TOKEN|>"
    assistant_prefix: str = "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    assistant_suffix: str = "<|END_OF_TURN_TOKEN|>"
    grammar: str = ""
    debug: bool = False

    @property
    def prefix(self):
        return {
            "system": self.system_prefix,
            "user": self.user_prefix,
            "assistant": self.assistant_prefix
        }
    
    @property
    def suffix(self):
        return {
            "system": self.system_suffix,
            "user": self.user_suffix,
            "assistant": self.assistant_suffix
        }

    def get_prompt_from_messages(self, messages, input_message, add_system=False):
        prompt = ""

        if add_system:
            prompt += self.prefix["system"] + self.system + self.suffix["system"]

        for message in messages:
            prompt += self.prefix[message["role"]] + message["content"] + self.suffix[message["role"]]
        prompt += self.prefix[input_message["role"]] + input_message["content"]

        return prompt