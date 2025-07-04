from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as se
import os
from dotenv import load_dotenv

load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# For LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "Question: {question}"),
    ]
)
print(prompt)

## Streamlit App
se.title("LangChain OpenAI Chatbot")
input_text = se.text_input("Enter your question: ")


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
    streaming=True,
    verbose=True,
)
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    se.write(chain.invoke({"question": input_text}))

