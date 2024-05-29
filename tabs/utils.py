# 会話を色分けする
def view(history, name_a="user", name_b="chatbot"):
    text = ""
    for user, chatbot in history:
        
        # 改行でspanが効かなくなっちゃうので、改行ごとにspanを挟む
        user_split_line = [name_a + ":"] + user.split("\n")
        user = "\n".join([f'<span style="color: navy">{line}</span>' for line in user_split_line])

        chatbot_split_line = [name_b + ":"] + chatbot.split("\n")
        chatbot = "\n".join([f'<span style="color: maroon">{line}</span>' for line in chatbot_split_line])

        text += f'{user}\n\n{chatbot}\n\n'

    return text