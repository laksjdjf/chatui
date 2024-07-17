import gradio as gr
from modules.llama_process import get_default_prompt, generate
import pandas as pd
import json

def questions_handler(format, csv_file, load_elyza, file_name):
    if load_elyza:
        from datasets import load_dataset
        ds = load_dataset("elyza/ELYZA-tasks-100")
        df = pd.DataFrame({"input":ds["test"]["input"]})
    else:
        df = pd.read_csv(csv_file)
    outputs = []
    total = len(df)
    for i in range(total):
        query = df.iloc[i]["input"]
        content = ""
        for chunk in generate(format.format(input=query)):
            content += chunk
        outputs.append(content)
        yield gr.update(value=f"進行中... {i+1}/{total}")
    else:
        df["output"] = outputs
        df.to_csv(f"output/{file_name}", index=False)
        yield gr.update(value=f"完了！{total}件のデータを出力しました。")

def eval_handler(format, preds_file, csv_file, load_elyza, file_name):
    if load_elyza:
        from datasets import load_dataset
        ds = load_dataset("elyza/ELYZA-tasks-100")
        df = pd.DataFrame({"input":ds["test"]["input"], "output":ds["test"]["output"], "eval_aspect":ds["test"]["eval_aspect"]})
    else:
        df = pd.read_csv(csv_file)
    preds = pd.read_csv(preds_file)
    df["pred"] = preds["output"]
    reasons = []
    grades = []
    total = len(df)
    
    for i in range(total):
        prompt = format.format(
            input_text=df.iloc[i]["input"],
            output_text=df.iloc[i]["output"],
            eval_aspect=df.iloc[i]["eval_aspect"],
            pred=df.iloc[i]["pred"]
        )
        content = ""
        for chunk in generate(prompt):
            content += chunk
        json_content = json.loads(content)
        reasons.append(json_content["reason"])
        grades.append(int(json_content["grade"]))
        yield gr.update(value=f"進行中... {i+1}/{total}, avg: {sum(grades)/len(grades):.2f}")
    else:
        df["reason"] = reasons
        df["grade"] = grades
        df.to_csv(f"output/{file_name}", index=False)
        yield gr.update(value=f"完了！{total}件のデータを出力しました。, avg: {sum(grades)/len(grades):.2f}")

def _get_default_prompt(add_system):
    return get_default_prompt("{input}", add_system)

FORMAT = """以下の問題に対する言語モデルによる回答を正解例及び評価基準に基づき評価し、評価理由および1,2,3,4,5の5段階評価による採点結果を「評価フォーマット」に示すようなJSON形式で返してください。

## 評価基準

点数についての基本的な評価基準は、以下のようになります。

### 基本的な評価基準

ベースとなる得点:

- **1点: 誤っている**
- **2点: 誤っているが、方向性は合っている**
- **3点: 部分的に誤っている, 部分的に合っている**
- **4点: 合っている**
- **5点: 役に立つ**

### 基本的な減点項目

ベースとなる得点から、以下のような要素を考慮して、得点を調整します。

- **不自然な日本語: -1点**
- **部分的なハルシネーション: -1点**
- **過度な安全性: 2点にする**
  - 「倫理的に答えられません」というような回答

### 問題ごとの評価基準
{eval_aspect}
  
## 問題
{input_text}

## 正解例
{output_text}

## 言語モデルによる回答
{pred}

## 評価フォーマット
{{"reason": "(採点基準に照らした評価内容)", "grade": (採点結果、1～5の5段階評価)}}"""

def _get_eval_default_prompt(add_system):
    return get_default_prompt(FORMAT, add_system)

def questions():
    with gr.Blocks() as questions_interface:
        gr.Markdown("問題を解かせるタブです。")
        with gr.Tab("Inference"):
            with gr.Row():
                with gr.Column():
                    format_textbox = gr.Textbox(
                        "{input}",
                        label="format",
                        placeholder="Enter your prompt here...",
                        interactive=True,
                        elem_classes=["prompt"],
                        lines=3,
                    )

                    add_system = gr.Checkbox(label="システムメッセージを追加", value=False)
                    
                    default_button = gr.Button("Default")
                    eval_button = gr.Button("Evaluation", variant="primary")
                    load_elyza = gr.Checkbox(label="Elyzaのデータをロード", value=False)
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
                questions_handler,
                inputs=[format_textbox, csv_file, load_elyza, file_name_textbox],
                outputs=[result]
            )

            default_button.click(
                _get_default_prompt,
                inputs=[add_system],
                outputs=[format_textbox],
            )
        
        with gr.Tab("Eval"):
            with gr.Row():
                with gr.Column():
                    eval_format_textbox = gr.Textbox(
                        "{input}",
                        label="format",
                        placeholder="Enter your prompt here...",
                        interactive=True,
                        elem_classes=["prompt"],
                        lines=3,
                    )

                    eval_add_system = gr.Checkbox(label="システムメッセージを追加", value=False)
                    
                    eval_default_button = gr.Button("Default")
                    eval_eval_button = gr.Button("Evaluation", variant="primary")
                    eval_preds_file = gr.File(label="Predictionsファイルをアップロード")
                    eval_load_elyza = gr.Checkbox(label="Elyzaのデータをロード", value=False)
                    eval_csv_file = gr.File(label="CSVファイルをアップロード")

                with gr.Column():
                    eval_file_name_textbox = gr.Textbox(
                        label="file_name",
                        value="output.csv",
                        placeholder="Enter your output file name here...",
                        interactive=True,
                    )
                    eval_result = gr.Textbox(label="Result", lines=5, interactive=False)
            
            eval_eval_button.click(
                eval_handler,
                inputs=[eval_format_textbox, eval_preds_file, eval_csv_file, eval_load_elyza, eval_file_name_textbox],
                outputs=[eval_result]
            )

            eval_default_button.click(
                _get_eval_default_prompt,
                inputs=[eval_add_system],
                outputs=[eval_format_textbox],
            )

    return questions_interface