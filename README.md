# Llama Local

GPUを使わずにLLama2 (ELYZA)のモデルで文書生成をするサンプルプログラム。

## Prepare environment

```
conda create -n llms python=3.10
conda activate llms
```

## Install libraries

### Langchineのインストール

```
pip install langchain
```

### LlamaCpp Pythonのインストール

```
pip install llama-cpp-python
```
または
```
CMAKE_ARGS="-DLLAMA_BLAS=ON" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

自分の環境（Ubuntu 22.04 LTS）では前者では動かなかった。(Kubuntu 22.04では前者で動いた)

GXXxxxのライブラリがないと怒られたときは次のコマンドを実行した解決した。

```
conda install -c conda-forge gxx_linux-64==11.1.0
```

### 言語モデルのダウンロード

```
wget -P models https://huggingface.co/mmnga/ELYZA-japanese-Llama-2-7b-fast-instruct-gguf/resolve/main/ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf

```

### 実行

次のコマンドを実行。
```
python elyza_cpu.py
```
またはStreamlitをインストールした上で次のコマンドを実行。
```
streamlit run webui.py
```
## Dockerでデプロイする

次のコマンドでデプロイできます。

```
docker compose up -d
```

Dockerイメージを作る際にELYZAのモデルをダウンロードするので最初のコンテナ起動には時間がかかります。

### 参考サイト

[https://python.langchain.com/docs/get_started/introduction](https://python.langchain.com/docs/get_started/introduction)

[https://python.langchain.com/docs/integrations/llms/llamacpp](https://python.langchain.com/docs/integrations/llms/llamacpp)

[https://llama-cpp-python.readthedocs.io/en/latest/](https://llama-cpp-python.readthedocs.io/en/latest/)


