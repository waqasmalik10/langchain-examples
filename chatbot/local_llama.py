from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# For LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

st.title('Local Llama Chatbot')
input_text = st.text_input("Ask a question:")


# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "Question: {question}"),
    ]
)

ollama = Ollama(
    model="llama2",
    temperature=0.7,
    top_p=0.9
)

output_parser = StrOutputParser()

chain = prompt | ollama | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))


