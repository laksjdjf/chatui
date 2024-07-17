import gradio as gr
import os
from modules.llama_process import load_model, clear_model, load_config, get_gbnf, load_logits_processor, PATTERNS
from modules.templates import get_template, template_list

def update_model_list(model_dir):
    return gr.update(choices=[model for model in os.listdir(model_dir) if model.endswith(".gguf")])

def setting(model_dir):
    with gr.Blocks() as setting_interface:
        with gr.Accordion("モデルのロード"):
            model_dir_state = gr.State(model_dir)
            with gr.Row():
                model_name = gr.Dropdown([model for model in os.listdir(model_dir) if model.endswith(".gguf")], label="model_name", scale=10)
                update_button = gr.Button(value="リスト更新🔄", variant="primary", scale=1)
            with gr.Row():
                if os.path.exists(os.path.join(model_dir,"mmprojs")):
                    mmprojs = ["None"] + [model for model in os.listdir(os.path.join(model_dir,"mmprojs")) if model.endswith(".gguf")]
                else:
                    mmprojs = ["None"]
                mmproj_name = gr.Dropdown(mmprojs, value="None", label="mmproj_name", scale=1)
                llava_handler = gr.Dropdown(["None", "Llava15ChatHandler", "Llava16ChatHandler"], label="llava_handler", scale=1, value="None")
            with gr.Row():
                if os.path.exists(os.path.join(model_dir,"loras")):
                    loras = ["None"] + [model for model in os.listdir(os.path.join(model_dir,"loras")) if model.endswith(".gguf")]
                else:
                    loras = ["None"]
                lora_name = gr.Dropdown(loras, value="None", label="lora_name", scale=1)
            ngl = gr.Slider(label="n_gpu_layers", minimum=0, maximum=256, step=1, value=256)
            ctx = gr.Slider(label="n_ctx", minimum=256, maximum=256000, step=256, value = 4096)
            ts = gr.Textbox(label="tensor_split")
            n_batch = gr.Slider(label="n_batch", minimum=32, maximum=4096, step=32, value=512)
            flash_attn = gr.Checkbox(label="flash_attn", value=True)
            no_kv_offload = gr.Checkbox(label="no_kv_offload", value=False)
            type_kv = gr.Dropdown(["q4_0", "q8_0", "f16"], value="f16", label="type_kv")
            logits_all = gr.Checkbox(label="logits_all", value=False)
            output_load_model = gr.Textbox(label="output", value="")
        
            with gr.Row():
                load_button = gr.Button(value="Load", variant="primary")
                clear_button = gr.Button(value="Clear", variant="secondary")

            update_button.click(
                update_model_list,
                inputs=[model_dir_state],
                outputs=[model_name],
            )

            load_button.click(
                load_model,
                inputs=[model_dir_state, model_name, mmproj_name, llava_handler, lora_name, ngl, ctx, ts, n_batch, flash_attn, no_kv_offload, type_kv, logits_all],
                outputs=[output_load_model],
            )

            clear_button.click(
                clear_model,
                outputs=[output_load_model],
            )

        with gr.Accordion("生成設定"):
            system = gr.Textbox(label="system", value="あなたは優秀なアシスタントです。", lines=2)
            temperature = gr.Slider(minimum=0, maximum=5, step=0.01, value=0.8, label="temperature")
            top_p = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.9, label="top_p")
            min_p = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.05, label="min_p")
            top_k = gr.Slider(minimum=1, maximum=256000, step=1, value=40, label="top_k")
            max_tokens = gr.Slider(minimum=1, maximum=256000, step=1, value=256, label="max_tokens")
            repeat_penalty = gr.Slider(minimum=1.0, maximum=2.0, step=0.01, value=1.0, label="repeat_penalty")
            with gr.Row():
                system_prefix = gr.Textbox(label="system_prefix", value="<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>", lines=2)
                system_suffix = gr.Textbox(label="system_suffix", value="<|END_OF_TURN_TOKEN|>", lines=2)
            with gr.Row():
                user_prefix = gr.Textbox(label="user_prefix", value="<|START_OF_TURN_TOKEN|><|USER_TOKEN|>", lines=2)
                user_suffix = gr.Textbox(label="user_suffix", value="<|END_OF_TURN_TOKEN|>", lines=2)
            with gr.Row():
                assistant_prefix = gr.Textbox(label="assistant_prefix", value="<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>", lines=2)
                assistant_suffix = gr.Textbox(label="assistant_suffix", value="<|END_OF_TURN_TOKEN|>", lines=2)
                                            
            grammar = gr.Textbox(label="grammar", value="", lines=3)
            debug = gr.Checkbox(label="debug", value=False)

            output_load_config = gr.Textbox(label="output", interactive=False)
            setting_list = [system, temperature, top_p, min_p, top_k, max_tokens, repeat_penalty, system_prefix, system_suffix, user_prefix, user_suffix, assistant_prefix, assistant_suffix, grammar, debug]

            template_dropdown = gr.Dropdown(template_list, label="template_list")
            grammar_dropdown = gr.Dropdown(["list", "json", "japanese", "problem_dict", "elyza_eval"], label="grammar_list")

        for setting in setting_list:
            setting.change(load_config, inputs=setting_list, outputs=output_load_config)
        template_dropdown.change(get_template, inputs=template_dropdown, outputs=[system_prefix, system_suffix, user_prefix, user_suffix, assistant_prefix, assistant_suffix]) 
        grammar_dropdown.change(get_gbnf, inputs=grammar_dropdown, outputs=[grammar])

        with gr.Accordion("Logits Processor"):
            languages_checkbox = gr.CheckboxGroup(list(PATTERNS.keys()), label="check the languages you want to ban")
            logits_processor_load_button = gr.Button("Load", variant="primary")
            logits_processor_output = gr.Textbox(label="number of ban tokens", interactive=False)
        
        logits_processor_load_button.click(
            load_logits_processor,
            inputs=[languages_checkbox],
            outputs=[logits_processor_output]
        )

    return setting_interface