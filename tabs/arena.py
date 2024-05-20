import gradio as gr
from tabs.setting import eval_output, get_prompt
import pandas as pd
import os

def calculate_elo(player1_rating, player2_rating, result1, k_factor=32):
    expected_score1 = 1 / (1 + 10 ** ((player2_rating - player1_rating) / 400))
    expected_score2 = 1 / (1 + 10 ** ((player1_rating - player2_rating) / 400))

    new_rating1 = player1_rating + k_factor * (result1 - expected_score1)
    result2 = 1 - result1
    new_rating2 = player2_rating + k_factor * (result2 - expected_score2)
    
    return new_rating1, new_rating2

# https://qiita.com/cabernet_rock/items/526d06a7993dfb61b75f
def Round_robin(N):

    match = []  # 試合の組み合わせを入れるリスト
    M=[i for i in range(N)]
    center = int(N/2)
    if N%2==0: # 偶数の時
        for i in range(N-1):
            match.append([(M[0+i],M[-1-i]) for i in range(center)])
            M = M[:1] + M[2:] + M[1:2] # 先頭を固定して１つずつずらす
    else: # 奇数の時
        for i in range(N):
            match.append([(M[i],M[-i+1]) for i in range(1,center+1)])
            M = M[1:] + M[:1] # １つずつずらす
    return match

def arena_handler(prompt_template, player, file_name):

    os.makedirs("output", exist_ok=True)

    players = [m.strip() for m in player.split(",")]
    outputs = ["A", "B"]

    num_players = len(players)
    
    ratings = [1500] * num_players
    wins = [0] * num_players
    draws = [0] * num_players
    losses = [0] * num_players
    num_games = [0] * num_players
    avg_prob = [0] * num_players

    round_robin = []
    _ = [round_robin.extend(robin) for robin in Round_robin(len(players))]

    for count, (i, j) in enumerate(round_robin):
        prompt_1 = prompt_template.format(player1=players[i], player2=players[j])
        likelihoods_1, log_likelihoods_1, _ = eval_output(prompt_1, outputs)
        prompt_2 = prompt_template.format(player1=players[j], player2=players[i])
        likelihoods_2, log_likelihoods_2, _ = eval_output(prompt_2, outputs)

        result_1 = likelihoods_1[0] > likelihoods_1[1]
        result_2 = likelihoods_2[0] < likelihoods_2[1]

        result = 0.5 if result_1 != result_2 else result_1
        prob_i = (likelihoods_1[0] + likelihoods_2[1]) / 2
        prob_j = (likelihoods_1[1] + likelihoods_2[0]) / 2

        ratings[i], ratings[j] = calculate_elo(ratings[i], ratings[j], result, k_factor=32)
        wins[i] += int(result)
        wins[j] += int(1 - result)
        draws[i] += 1 if result == 0.5 else 0
        draws[j] += 1 if result == 0.5 else 0
        losses[i] += int(1 - result)
        losses[j] += int(result)
        num_games[i] += 1
        num_games[j] += 1
        avg_prob[i] = prob_i / num_games[i] + avg_prob[i] * (num_games[i] - 1) / num_games[i]
        avg_prob[j] = prob_j / num_games[j] + avg_prob[j] * (num_games[j] - 1) / num_games[j]

        label = {player+ "/" + str(int(rating)):rating / 2000 for player, rating in zip(players, ratings)}

        yield gr.update(value=f"{count+1}/{len(round_robin)}"), gr.update(value=label), None
    else:
        df = pd.DataFrame(
            {
                "Player": players,
                "Rating": ratings,
                "Win": wins,
                "Loss": losses,
                "Draw": draws,
                "Avg Prob": avg_prob,
            }
        ).sort_values("Rating", ascending=False)

        df.to_csv(f"output/{file_name}")
        yield gr.update(value=f"完了！"), gr.update(value=label), gr.update(value=df)

def update_prompt(post_prompt=None):
    prompt = get_prompt("<input>", post_prompt)
    return gr.update(value=prompt, autoscroll=True)

def arena():
    with gr.Blocks() as arena_interface:
        gr.Markdown("テキスト評価用タブです。")

        with gr.Row():
            with gr.Column():
                prompt_textbox = gr.Textbox(
                    label="Input",
                    placeholder="Enter your prompt here...",
                    interactive=True,
                    elem_classes=["prompt"],
                    lines=3,
                )

                player_textbox = gr.Textbox(
                    label="put player name here (a, b, c, d, e...)",
                    lines=2,
                    interactive=True,
                )
                output_textbox = gr.Textbox(
                    label="Output",
                    value="output.csv",
                    placeholder="Enter your output file name here...",
                    interactive=True,
                )
                
                default_button = gr.Button("Default")
                eval_button = gr.Button("Evaluation", variant="primary")

            with gr.Column():
                progress = gr.Textbox(label="Progress", value="0/0")
                label = gr.Label("Result", num_top_classes=30)
                df = gr.Dataframe(
                    headers=["Player", "Rating", "Win", "Loss", "Draw", "Avg Prob"],
                    datatype=["str", "number", "number", "number", "number", "number"],
                    interactive=False,
                )
        

        eval_button.click(
            arena_handler,
            inputs=[prompt_textbox, player_textbox, output_textbox],
            outputs=[progress, label, df]
        )

        default_button.click(
            update_prompt,
            outputs=[prompt_textbox],
        )

    return arena_interface