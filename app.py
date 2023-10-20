import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from azure.storage.blob import BlobServiceClient
import os

from io import BytesIO

def get_pdfs_from_blob(account_url, container_name, sas_token):
    account_url = "https://stknowledgebasedata.blob.core.windows.net/"
    #st.write(f"Account URL TEST2: {account_url}")
    blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)
    container_client = blob_service_client.get_container_client(container_name)
    blobs = container_client.list_blobs()

    pdf_data = []
    for blob in blobs:
        blob_client = container_client.get_blob_client(blob.name)
        pdf_data.append(blob_client.download_blob().readall())
    return pdf_data


def get_pdf_text(pdf_data_list):
    text = ""
    for pdf_data in pdf_data_list:
        pdf_stream = BytesIO(pdf_data)
        pdf_reader = PdfReader(pdf_stream)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="BİRİ with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Azure Blob Storage bilgilerini .env dosyasından alın
    ACCOUNT_URL = os.getenv("ACCOUNT_URL") 
    CONTAINER_NAME = os.getenv("CONTAINER_NAME") 
    SAS_TOKEN = os.getenv("SAS_TOKEN") 

    # Azure Blob Storage'dan PDF'leri çekin
    pdf_data_list = get_pdfs_from_blob(ACCOUNT_URL, CONTAINER_NAME, SAS_TOKEN)

    # PDF datalarını PdfReader ile okuyarak metne dönüştürün
    raw_text = get_pdf_text(pdf_data_list)

    # get the text chunks
    text_chunks = get_text_chunks(raw_text)

    # create vector store
    vectorstore = get_vectorstore(text_chunks)

    # create conversation chain
    st.session_state.conversation = get_conversation_chain(vectorstore)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("BİRİ with multiple PDFs :books:")
    user_question = st.text_input("Dokümanın ile ilgili soru sor:")
    if user_question:
        handle_userinput(user_question)


if __name__ == '__main__':
    main()
