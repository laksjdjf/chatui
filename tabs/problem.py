import gradio as gr
from modules.llama_process import eval_choice, get_prompt_from_messages, get_default_prompt, generate
import pandas as pd
import numpy as np
import json

DEFAULT_PROMPT = """以下の問題の答えを選択肢から選んで記号のみを答えてください。
分類:{category}
問題:{problem}
選択肢:
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}
"""

DEFAULT_PROMPT_DICT = """以下の問題の答えを選択肢から選んで回答理由と答えの記号をそれぞれreason, answerをkeyとした辞書形式で回答してください。回答理由は詳細にしてください。
分類:{category}
問題:{problem}
選択肢:
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}
"""

def default_prompt(dict_mode):
    if dict_mode:
        return DEFAULT_PROMPT_DICT
    else:
        return DEFAULT_PROMPT

def problem_handler(dict_mode, prompt, csv_file, file_name):

    df = pd.read_csv(csv_file)
    prob_df = df[["category", "problem", "A", "B", "C", "D", "E"]]
    ans_df = df[["正答", "大誤答"]]
    answers = []
    reasons = []

    result = ""
    collect = 0
    daigoto = 0

    for i in range(len(df)):
        prob = prob_df.iloc[i].to_dict()
        prompt_target = prompt.format(**prob)
        
        if dict_mode:
            content = ""
            for chunk in generate(prompt_target):
                content += chunk
            try:
                ret_dict = json.loads(content)
                ans = ret_dict["answer"]
                reason = ret_dict["reason"]
                ans_str = prob[ans]
            except:
                ans = "X"
                ans_str = "形式ミス"
                reason = content

            answers.append(ans)
            reasons.append(reason)
        else:
            ans = eval_choice(prompt_target, ["A", "B", "C", "D", "E"])
            ans_str = prob[ans]
            answers.append(ans)
        
        if ans == ans_df.iloc[i]["正答"]:
            res = "正解"
            collect += 1
        elif ans == ans_df.iloc[i]["大誤答"]:
            res = "大誤答"
            daigoto += 1
        else:
            res = "不正解"
        result += f"問題{i+1}: {prob['problem']} \n 回答： {ans_str} \n 結果： {res} \n\n"
        collect_result = f"\n問題数:{i+1}, 正答数:{collect}, 正答率:{collect/(i+1):.2%}, 大誤答数:{daigoto}"
        yield gr.update(value=result + collect_result, autoscroll=True, interactive=True)
    else:
        if dict_mode:
            pd.DataFrame({"answer": answers, "reason": reasons}).to_csv(f"output/{file_name}", index=False)
        else:
            pd.Series(answers).to_csv(f"output/{file_name}", index=False)
        yield gr.update(value=result + collect_result, autoscroll=True, interactive=False)

def problem():
    with gr.Blocks() as problem_interface:
        gr.Markdown("問題を解かせるタブです。")

        with gr.Row():
            with gr.Column():
                prompt_textbox = gr.Textbox(
                    label="Input",
                    placeholder="Enter your prompt here...",
                    interactive=True,
                    elem_classes=["prompt"],
                    lines=3,
                )
                
                default_button = gr.Button("Default")
                eval_button = gr.Button("Evaluation", variant="primary")

                user_textbox = gr.Textbox(label="input for default button", value=DEFAULT_PROMPT, lines=2)
                dict_mode = gr.Checkbox(label="辞書モード", value=False)
                csv_file = gr.File(label="CSVファイルをアップロード")

            with gr.Column():
                file_name_textbox = gr.Textbox(
                    label="file_name",
                    value="output.csv",
                    placeholder="Enter your output file name here...",
                    interactive=True,
                )
                result = gr.Textbox(label="Result", lines=5, interactive=False)
        
        eval_button.click(
            problem_handler,
            inputs=[dict_mode, prompt_textbox, csv_file, file_name_textbox],
            outputs=[result]
        )

        dict_mode.change(
            default_prompt,
            inputs=[dict_mode],
            outputs=[user_textbox],
        )

        default_button.click(
            get_default_prompt,
            inputs=[user_textbox],
            outputs=[prompt_textbox],
        )

    return problem_interface