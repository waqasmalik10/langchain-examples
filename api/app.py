from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

from langserve import add_routes

from fastapi import FastAPI
import uvicorn
import os

from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A Simple API Server"
)

openai_model = ChatOpenAI()
deepseek_model = Ollama(
    model="deepseek-r1:7b"
)

prompt1 = ChatPromptTemplate.from_template(
    "write me an essay about {topic} with 100 words."
)
prompt2 = ChatPromptTemplate.from_template(
    "write me a poem about {topic} with 100 words"
)


add_routes(
    app, 
    ChatOpenAI(),
    path="/openai"
)
add_routes(
    app, 
    prompt1 | openai_model,
    path="/essay"
)
add_routes(
    app, 
    prompt2 | deepseek_model,
    path="/poem"
)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)