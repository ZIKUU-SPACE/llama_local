import threading
import queue
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from langchain.llms import LlamaCpp
#from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# from langchain.schema import (
#     HumanMessage,
#     SystemMessage,
#     AIMessage
# )

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"


app = FastAPI(
    title="LangChain",
)

class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration:
            raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)

class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        self.gen.send(token)

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

def make_prompt(message):
    prompt = "{b_inst} {system}{prompt} {e_inst} ".format(
        b_inst=B_INST,
        system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
        prompt=message,
        e_inst=E_INST,
    )
    return prompt


def llm_thread(g, prompt):
    try:
        chat = load_model()
        chat(make_prompt(prompt))

    finally:
        g.close()


def chat(prompt):
    g = ThreadedGenerator()
    threading.Thread(target=llm_thread, args=(g, prompt)).start()
    return g


class Message(BaseModel):
    message: str


@app.post("/generate")
async def generate(message: Message):
    return StreamingResponse(
        chat(message.message),
        media_type='text/event-stream'
    )
    
def start():
    uvicorn.run("fastapi_streaming:app", port=8000, reload=True)


if __name__ == "__main__":
    start()

