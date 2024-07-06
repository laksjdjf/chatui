# chatui

[gradio](https://www.gradio.app/)および[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)を利用したUIです。変な機能しかありません。

# 起動方法
gradio(4.37.2 or 4.29.0推奨)とllama-cpp-pythonを好きにインストールしてください。
RAGタブを利用する場合はfaiss-cpu（とPDFを入れる場合はpdfminer)が必要です。
```
pip install gradio==4.37.2
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<cuda-version>

[Optional]
pip install faiss-cpu pdfminer.six
```

ggufファイルがあるディレクトリを指定して起動します。

```
python main.py -m <gguf-directory>
```

※`--share`でshareできます。

# 各タブの説明
## Setting
一番右側にあります。モデルのロード及び生成に関する各種設定を行います。

モデルはggufファイルを設定し、ロード設定を変更したのちロードボタンを押してください。

生成設定は、各項目を変更するだけで自動的に設定できます。特にpromptのテンプレートはtemplate_listから適切なものを選ぶか自分で設定してください。

またlogits-processorでは禁止する言語を選べます（うまくいってない部分もあるかも）。禁止したい言語だけ選択し、ロードボタンを教えてください。

## Chat
チャットタブです。

システムプロンプトはSettingタブにあるものではなく、チャット設定のsystemに従います。

## Completion
自由な入力ができるタブです。結果はEOSを除いて入力欄に直接追加されます。

user部分に何らかの質問をし、Defaultボタンを押すとチャットテンプレートが自動的に作成されます。

Undoは一個前までしか戻せません。ーｑ－

## likelihood
入力と出力を与えて、出力の尤度を計算するタブです。

**利用する場合は、モデルロード時にlogits_allのチェックをつけてください。**

Defaultボタンでinput for default buttonの内容からチャットテンプレートが作成されます。

## EvalSentence
与えられた文章のperplexityを計算します。また予測確率が高いトークンは赤、低いトークンは青で表示されます。
**利用する場合は、モデルロード時にlogits_allのチェックをつけてください。**

## Problem
AIに選択問題を与えて回答させます。

category, problem, A, B, C, D, E, 正解, 大誤答のcsvファイルを入力することで選択問題を解かせることができます。

正解及び大誤答はラベル（AとかB）です。
## Arena
AIに単語をいっぱい与えて、総当たりで競わせます。

カンマ区切りで単語列を与えることでランキングがつくれます。

## RAG
なんちゃってRAGができます。

Embedding Model欄でテキストファイルもしくはPDF（pdfminer install時のみ）を与えて、ロードを押すことでRAGができます。

Number of AugmentationsはAIに与えるChunkの数になります。

[intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)以外では動かないと思います。

## AI2AI
AI同士の会話を再現します。

二つのシステムプロンプトを与えるとA→Bの順番で話し出します。

Aに対しては最初にfirst_messageが与えられます。あまり意味のない会話を促すようなものを入力するとよいでしょう。

Number of completionで会話の往復回数を設定できます。



