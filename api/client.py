import requests
import streamlit as st

def get_openai_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke", json={"input": {"topic": input_text} } )
    return response.json()['output']['content']


def get_deepseek_response(input_text):
    response = requests.post('http://localhost:8000/poem/invoke', json={ "input": {'topic': input_text} })
    return response.json()['output']

st.title('Langchain Demo')
input_text = st.text_input("Write an eassy on")
input_text2 = st.text_input("Write a poem on")

if input_text:
    st.write( get_openai_response(input_text) )
if input_text2:
    st.write( get_deepseek_response(input_text2) )