import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from PIL import Image
from tablaGo import update_excel

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
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
    
    if "button_clicked" not in st.session_state:
        st.session_state.button_clicked = False
    if "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = False
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.set_page_config(page_title="IA Chat", page_icon=Image.open('img\\proyeco_logo.jpg'))
    st.write(css, unsafe_allow_html=True)
    
    st.header("Pregunta sobre tu PDF")
    user_question = st.text_input(label = "Texto", disabled=not st.session_state.button_clicked, placeholder = "Escribe aquí")

 
    if user_question is not None:
        handle_userinput(user_question)


    with st.sidebar:
        st.subheader("Documentos")
        pdf_docs = st.file_uploader("Sube aquí tus archivos y presiona 'Procesar'", accept_multiple_files=True)

        if pdf_docs:
            st.session_state.file_uploaded = True
        else:
            st.session_state.file_uploaded = False
        
        if st.button("Procesar"):
            if st.session_state.file_uploaded:
                st.session_state.button_clicked = True
                st.experimental_rerun()  # Rerun the script to update the UI
            else:
                st.warning("Por favor suba un documento antes de procesar")

        if st.session_state.file_uploaded and st.session_state.button_clicked:
            with st.spinner("Procesando"):
                # Get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

        if st.button("Generar Ficha Go", help="Subir primero el archivo pdf para poder generar la ficha", disabled=not st.session_state.button_clicked):
            st.write("ole")
            st.download_button(update_excel())

if __name__ == '__main__':
    main()