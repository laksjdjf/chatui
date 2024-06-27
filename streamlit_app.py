import streamlit as st
import os
import sys
from llama_cpp import Llama

model_dir = sys.argv[1]
USER_NAME = "user"
ASSISTANT_NAME = "assistant"

def load_model(model_name, ngl=256, ctx=512, ts=None, n_batch=512, flash_attn=True):
    model_path = os.path.join(model_dir, model_name)
    ts = [float(x) for x in ts.split(",")] if ts else None
    model = Llama(
        model_path=model_path,
        n_gpu_layers=ngl,
        n_batch=n_batch,
        tensor_split=ts,
        n_ctx=ctx,
        flash_attn=flash_attn
    )
    return model

def generate(model, prompt, temperature, top_p, max_tokens, repeat_penalty, debug):
    if debug:
        print(f"Prompt: {prompt}")

    for chunk in model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        grammar=None,
        logits_processor=None,
        stream=True,
    ):
        yield chunk["choices"][0]["text"]

def get_prompt_from_history(history, system_template, user_template, chatbot_template):
    prompt = system_template.format(system=system)
    for i, data in enumerate(history):
        if i == len(history) - 1:
            if data["role"] == USER_NAME:
                prompt += user_template.split("{user}")[0] + data["content"]
            elif data["role"] == ASSISTANT_NAME:
                prompt += chatbot_template.split("{chatbot}")[0] + data["content"]
        else:
            if data["role"] == USER_NAME:
                prompt += user_template.format(user=data["content"])
            elif data["role"] == ASSISTANT_NAME:
                prompt += chatbot_template.format(chatbot=data["content"])
    return prompt

if __name__ == "__main__":
    with st.sidebar:
        st.image("https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png", use_column_width=True)
        tab_model, tab_setting = st.tabs(["Model", "Setting"])
        with tab_model:
            ## model load
            model_files = os.listdir(model_dir)
            selected_model = st.selectbox("Choose a model to load", model_files)
            ngl = st.slider("n_gpu_layers", min_value=0, max_value=256, step=1, value=256)
            ctx = st.slider("n_ctx", min_value=256, max_value=256000, step=256, value=4096)
            ts = st.text_input("tensor_split", value="")
            n_batch = st.slider("n_batch", min_value=32, max_value=4096, step=32, value=512)
            flash_attn = st.checkbox("flash_attn", value=True)
            no_kv_offload = st.checkbox("no_kv_offload", value=False)

            if st.button("Load Model", type="primary"):
                st.session_state.model = load_model(selected_model, ngl, ctx, ts, n_batch, flash_attn)
                st.session_state.model_name = selected_model
                st.success(f"Model '{selected_model}' loaded successfully!")

        with tab_setting:
            system = st.text_input("system", value="あなたは優秀なアシスタントです。")
            temperature = st.slider("temperature", min_value=0.0, max_value=5.0, step=0.1, value=0.8)
            top_p = st.slider("top_p", min_value=0.0, max_value=1.0, step=0.01, value=0.9)
            max_tokens = st.slider("max_tokens", min_value=32, max_value=4096, step=32, value=256)
            repeat_penalty = st.slider("repeat_penalty", min_value=0.0, max_value=1.0, step=0.01, value=1.0)
            system_template = st.text_area("system_template", value="<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system}<|END_OF_TURN_TOKEN|>")
            user_template = st.text_area("user_template", value="<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user}<|END_OF_TURN_TOKEN|>")
            chatbot_template = st.text_area("chatbot_template", value="<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{chatbot}<|END_OF_TURN_TOKEN|>")
            debug = st.checkbox("debug", value=False)

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = -1
    if "edit_msg" not in st.session_state:
        st.session_state.edit_msg = ""

    if "model" not in st.session_state or st.session_state.model is None:
        st.markdown("Please load a model first.")
    else:
        st.markdown(f"{st.session_state.model_name} loaded.")

    i = 0
    for i, chat in enumerate(st.session_state.chat_log):
        col1, col2 = st.columns([9, 1])
        with col1:
            if st.session_state.edit_mode == i:
                st.session_state.edit_msg = st.text_area("Edit message", value=st.session_state.edit_msg, key=f"edit_{i}")
            else:
                messages = st.chat_message(chat["role"])
                messages.write(chat["content"])
        with col2:
            if st.session_state.edit_mode == i:
                if st.button("Save", key=f"save_{i}", type="primary"):
                    st.session_state.chat_log[i]["content"] = st.session_state.edit_msg
                    st.session_state.edit_mode = -1
                    st.experimental_rerun()
                if st.button("Delate", key=f"delete_{i}"):
                    st.session_state.chat_log.pop(i)
                    st.session_state.edit_mode = -1
                    st.experimental_rerun()
                if st.button("Cancel", key=f"cancel_{i}"):
                    st.session_state.edit_mode = -1
                    st.experimental_rerun()
            else:
                if st.button("Edit", key=f"edit_{i}"):
                    st.session_state.edit_mode = i
                    st.session_state.edit_msg = chat["content"]
                    st.experimental_rerun()
                    
    user_msg = st.chat_input("ここにメッセージを入力")
    
    if user_msg:
        with st.chat_message(USER_NAME):
            st.empty()
            st.empty()
            st.write(user_msg)
        st.session_state.chat_log.append({"role": USER_NAME, "content": user_msg})
        
        messages = st.chat_message(ASSISTANT_NAME)
        st.session_state.chat_log.append({"role": ASSISTANT_NAME, "content": ""})
        prompt = get_prompt_from_history(st.session_state.chat_log, system_template, user_template, chatbot_template)
        content = st.session_state.chat_log[-1]["content"]
        with messages:
            msg = st.empty()
            for chunk in generate(st.session_state.model, prompt, temperature, top_p, max_tokens, repeat_penalty, debug):
                content += chunk
                msg.write(content)
        st.session_state.chat_log[-1]["content"] = content
        st.experimental_rerun()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Add User"):
            st.session_state.chat_log.append({"role": USER_NAME, "content": ""})
            st.experimental_rerun()
    with col2:
        if st.button("Add Assistant"):
            st.session_state.chat_log.append({"role": ASSISTANT_NAME, "content": ""})
            st.experimental_rerun()
    with col3:
        if st.button("Clear Chat Log"):
            st.session_state.chat_log = []
            st.experimental_rerun()
    if st.button("Continue", type="primary"):
        prompt = get_prompt_from_history(st.session_state.chat_log, system_template, user_template, chatbot_template)
        content = st.session_state.chat_log[-1]["content"]
        with messages:
            msg = st.empty()
            for chunk in generate(st.session_state.model, prompt, temperature, top_p, max_tokens, repeat_penalty, debug):
                content += chunk
                msg.write(content)
        st.session_state.chat_log[-1]["content"] = content
        st.experimental_rerun()
