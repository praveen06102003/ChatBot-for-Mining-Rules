import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os

# Ensure the required package is installed
os.system('pip install sentence-transformers')

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
llm = ChatGroq(
    groq_api_key='',
    model_name="llama3-70b-8192"
)
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context and also the information present in the llm about mining.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("ChatBot for Mining Rules")
with st.spinner("Loading pdf"):
    pdf_loader = PyPDFLoader("Mining_Rules.pdf")
    pdf_docs = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pdf_final_documents = text_splitter.split_documents(pdf_docs)
    pdf_vectors = FAISS.from_documents(pdf_final_documents, embeddings)
document_chain = create_stuff_documents_chain(llm, prompt)
pdf_retriever = pdf_vectors.as_retriever()
pdf_retrieval_chain = create_retrieval_chain(pdf_retriever, document_chain)

pdf_prompt = st.chat_input("Enter the prompt to search")
if pdf_prompt:
    with st.spinner("Generating the response"):
        try:
            st.session_state.messages.append({'role': 'user', 'content': pdf_prompt})
            response = pdf_retrieval_chain.invoke({"input": pdf_prompt})
            st.session_state.messages.append({'role': 'assistant', 'content': response['answer']})
        except Exception as e:
            st.error(f"An error occurred: {str(e)}. Please try again later or contact support.")

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])
