from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

def create_client():
        return ChatAnthropic(
                model_name=os.getenv("MODEL_ID"),
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                base_url=os.getenv("ANTHROPIC_BASE_URL"),
                max_tokens_to_sample=1024
        )

llm = create_client()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Genzy egoestic assistant who uses slang words."),
    ("placeholder", "{history}"),
    ("human", "{input}")
])

parser = StrOutputParser()

chain = prompt | llm | parser


stores = {}
def get_history(session_id:str):
    if session_id not in stores:
        stores[session_id] =  ChatMessageHistory()
    return stores[session_id]

chatbot = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)


class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    content: str

@app.post("/chat", response_model=ChatResponse)
def post_chat(request: ChatRequest):
    config = {
    "configurable":{
        "session_id": request.userId
        }
    }
    response = chatbot.invoke({"input":request.message}, config=config)
    return ChatResponse(content=response)
    

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    config = {"configurable": {"session_id": request.session_id}}

    def event_generator():
        for chunk in chatbot.stream({"input": request.message}, config=config):
            yield chunk  # chunk is already a string, thanks to StrOutputParser

    return StreamingResponse(event_generator(), media_type="text/plain")