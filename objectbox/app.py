import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"]
groq_api_key = os.getenv["GROQ_API_KEY"]

st.title("Objectbox Vector Store")


## Vector Embeddings and Object Vectorestore
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census") ## Daat Ingestion
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.final_documents = st.session_state.text_splitter.split_documents( st.session_state.docs[:20] )

        st.session.vectors = ObjectBox.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings,
            persist_directory="./objectbox_db"
        )


llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only. 
    Do not use any other information.
    Context: {context}
    Question: {input}
    """
) 

input_prompt = st.text_input("Enter your question from documents:")

if st.button("Document Embeddings"):
    vector_embedding()
    st.write("ObjectBox Vector Store Created")


if input_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({"input":input_prompt})
    print("Response Time: " , time.process_time - start)

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks 
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------")