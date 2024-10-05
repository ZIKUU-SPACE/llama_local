import streamlit as st
from langchain.callbacks.base import BaseCallbackManager
from langchain.llms import LlamaCpp
from langchain.callbacks.base import BaseCallbackHandler
import logging

logging.basicConfig(filename='streamlit.log')

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。どんな難問にも挑戦してくれると信じています。"

#model_path = "models/ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf" '''
model_path = "models/ELYZA-japanese-Llama-2-13b-fast-instruct-q4_K_M.gguf"

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")


def make_prompt(message):
    prompt = "{b_inst} {system}{prompt} {e_inst} ".format(
        b_inst=B_INST,
        system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
        prompt=message,
        e_inst=E_INST,
    )
    return prompt


with st.sidebar:
    st.header('日本語生成AI')
    st.subheader('作る・学ぶ モノづくり塾『ZIKUU』')
    st.caption('このプログラムはELYZA社による70億パラメーターの大規模言語モデルを使用してCPUのみで文章を生成するためにLlamaCppを利用しています。')
    st.divider()

    st.text("設定")
    max_tokens = st.slider('Max Tokens', value=4096, min_value=1024, max_value=8192, step=16)
    temperature = st.slider('Temperature', value=0.75, min_value=0.1, max_value=1.5, step=0.05)

with st.form(key="generation_form"):
    prompt = st.text_area('質問')
    do_generate = st.form_submit_button('送信')
    if do_generate and prompt:
        with st.spinner("生成中..."):
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box, display_method='write')
            chat = LlamaCpp(
                model_path=model_path,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                callback_manager=BaseCallbackManager([stream_handler]), 
                verbose=False
            )
            res = chat(make_prompt(prompt))
