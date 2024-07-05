import shutil

def view(history, name_a, name_b, icon_a=None, icon_b=None, system_a=None, system_b=None):
    if icon_a:
        icon_a_copy = "tmp/icon_a.png"
        shutil.copy(icon_a, icon_a_copy)
        name_a = f'<img src="/file/{icon_a_copy}" width="30">'
        print(name_a)
    else:
        name_a += ":"
    if icon_b:
        icon_b_copy = "tmp/icon_b.png"
        shutil.copy(icon_b, icon_b_copy)
        name_b = f'<img src="/file/{icon_b_copy}" width="30">'
    else:
        name_b += ":"

    if system_b is None:
        text = f"system= {system_a}\n\n"
    else:
        text = f"{name_a}system= {system_a}\n\n{name_b}system= {system_b}\n\n"
        
    for user, chatbot in history:
        
        # 改行でspanが効かなくなっちゃうので、改行ごとにspanを挟む
        user_split_line = [name_a] + user.split("\n")
        user = "\n".join([f'<span style="color: navy">{line}</span>' for line in user_split_line])

        chatbot_split_line = [name_b] + chatbot.split("\n")
        chatbot = "\n".join([f'<span style="color: maroon">{line}</span>' for line in chatbot_split_line])

        text += f'{user}\n\n{chatbot}\n\n'

    return text