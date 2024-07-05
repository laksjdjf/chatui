# chatui

[gradio](https://www.gradio.app/)および[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)を利用したUIです。変な機能しかありません。

# 起動方法
gradio(4.37.2推奨)とllama-cpp-pythonを好きにインストールしてください。
RAGタブを利用する場合はfaiss-cpu（とPDFを入れる場合はpdfminer)が必要です。
```
pip install gradio==4.37.2
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<cuda-version>

[Optional]
pip install faiss-cpu pdfminer.six
```

ggufファイルがあるディレクトリを指定して起動します。

`python main.py -m <gguf-directory>`

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

