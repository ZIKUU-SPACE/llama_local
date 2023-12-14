from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

""" モデルのパスを間違えずに入力"""
model_path = "models/ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf"

def load_model() -> LlamaCpp:
    """Load model"""
    callback_manager: CallbackManager = CallbackManager([StreamingStdOutCallbackHandler()])

    model: LlamaCpp = LlamaCpp(
        model_path=model_path,
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        callback_manager=callback_manager, 
        verbose=True
    )

    return model


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"


while True:
    message = input("\n質問: ")

    prompt = "{b_inst} {system}{prompt} {e_inst} ".format(
        b_inst=B_INST,
        system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
        prompt=message,
        e_inst=E_INST,
    )

    print("\n回答: ")
    llm = load_model()

    llm(prompt)