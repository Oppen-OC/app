import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from PIL import Image
from tablaGo import tablaGo
import json
from pdfminer.high_level import extract_text
from io import BytesIO
from pdf2image import convert_from_bytes
import pytesseract
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage



embeddings = None
store = {}
pdf_docs = None

def initiate_llm():
    # Load the configuration from the JSON file
    with open('config.json') as f:
        config = json.load(f)

    # Extract model parameters from the configuration
    model_params = config['model']['parameters']

    # Define system message and user instructions
    system_message = config['system_messages']['startup_message']
    user_instruction = config['user_instructions']['init_prompt']

    # Create an instance of ChatOpenAI with the extracted parameters
    llm = ChatOpenAI(
        model_name=config['model']['version'],
        temperature=model_params['temperature'],
        max_tokens=model_params['max_tokens'],
        top_p=model_params['top_p'],
        frequency_penalty=model_params['frequency_penalty'],
        presence_penalty=model_params['presence_penalty'],
        extra_body=model_params  # Include extra parameters if needed
    )

    return llm


def get_pdf_text(pdf_docs):
    text = ""  # Initialize the text variable

    for pdf in pdf_docs:
        try:
            pdf_bytes = pdf.getvalue()
        except AttributeError:
            continue  # Skip if pdf does not have getvalue()

        # Create a PDF file-like object from bytes
        pdf_file_like = BytesIO(pdf_bytes)

        # Extract text using pdfminer.six high-level API
        try:
            pdf_text = extract_text(pdf_file_like)
            if pdf_text.strip():  # Check if extracted text is not empty
                text += pdf_text
            else:
                raise ValueError("No text extracted, falling back to OCR")
        except Exception as e:
            print(f"Error extracting text: {e}")
            # Convert PDF to images for OCR
            try:
                images = convert_from_bytes(pdf_bytes)
                print("imagen a bits")
                for image in images:
                    ocr_text = pytesseract.image_to_string(image)
                    text += ocr_text
                    print("Textos")
            except Exception as ocr_e:
                print(f"Error performing OCR: {ocr_e}")
                continue  # Skips if OCR fails

    if text:
        # Write the extracted text to a file
        with open("Output.txt", "w", encoding="utf-8") as text_file:
            text_file.write(text)

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
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print(vectorstore.index.ntotal)
    vectorstore.save_local("ser")
    return vectorstore

def get_vectorstore_local():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.load_local("ser", embeddings, allow_dangerous_deserialization=True)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    print("Got session history")
    return store[session_id]

llm = initiate_llm()
with_message_history = RunnableWithMessageHistory(llm, get_session_history)

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        if st.session_state.chat_history and len(st.session_state.chat_history) > 2:
            st.error("Conversation object is None. Cannot handle user input.")
        return
        
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        print("Conversacion subida")
            
    except AssertionError as e:
        st.error(f"AssertionError: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
            
        #if len(st.session_state.chat_history) > 1:
    for i, message in enumerate(st.session_state.chat_history[-2:]):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    return

def handle_userinput2(user_question):
    if st.session_state.conversation is None:
        if st.session_state.chat_history and len(st.session_state.chat_history) > 2:
            st.error("Conversation object is None. Cannot handle user input.")
        return
        
    try:
        response = with_message_history.invoke(
            [HumanMessage(content=user_question)],
            config={"configurable": {"session_id": "chat_history"}},
            )
        #st.session_state.conversation({'question': user_question})
        #st.session_state.chat_history = response.content
        print("Respuesta enviada")
            
    except AssertionError as e:
        st.error(f"AssertionError: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
            
        #if len(st.session_state.chat_history) > 1:
    for i, message in enumerate(st.session_state.chat_history[-2:]):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", response.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", response.content), unsafe_allow_html=True)
    return

def main():
    load_dotenv()

    tg = tablaGo("ficha.xlsx",
                 [4, 5, 6, 12, 13, 15, 19, 21, 23, 24, 26, 28, 30, 32],
                 "prompts.txt") 

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
    if "ficha" not in st.session_state:
        st.session_state.ficha = None
    if "vectorstore" not in st.session_state:
        st.session_state.vector_store = None

    st.set_page_config(page_title="IA Chat", page_icon=Image.open('img\\proyeco_logo.jpg'))
    st.write(css, unsafe_allow_html=True)

    st.header("Pregunta sobre tu PDF")
    user_question = st.text_input(label="Texto", disabled=not st.session_state.button_clicked, placeholder="Escribe aquí",key='widget')

    if len(user_question) > 1:
        handle_userinput2(user_question)

    with st.sidebar:
        st.subheader("Documentos")
        pdf_docs = st.file_uploader("Sube aquí tus archivos y presiona 'Procesar'", accept_multiple_files=True, type  = "pdf")

        if pdf_docs:
            st.session_state.file_uploaded = True
        else:
            st.session_state.file_uploaded = False

        if st.button("Procesar"):
            if st.session_state.file_uploaded:
                st.session_state.button_clicked = True
                print("Procesar presionado")
                if st.session_state.vector_store is None:
                    with st.spinner("Procesando"):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        #st.session_state.vector_store = get_vectorstore(text_chunks)
                        st.session_state.vector_store = get_vectorstore_local()
                        print(st.session_state.conversation)
                        #st.session_state.conversation = get_conversation_chain(st.session_state.vector_store)
                        print("GOT THE VECTOR STORE")
                st.rerun()

            else:
                st.warning("Por favor suba un documento antes de procesar")
                print("FILE UPLOADED IS FALSE")

        if st.button("Generar Ficha Go", 
                     help="Subir primero el archivo pdf para poder generar la ficha", 
                     disabled=not st.session_state.button_clicked):
            
            #Accede al documento de querys y llama al update_excel
            st.session_state.ficha = st.session_state.conversation
            
            st.session_state.chat_history = st.session_state.ficha
            
            #Para el excel de preguntas
            for query in tg.prompts:
                response = st.session_state.ficha({'question': query})['answer']
                result = tg.update_excel("ficha.xlsx", "A1 Resumen", response)
            
            if result is not None:  # Ensure result is appropriate for download_button
               st.download_button(
                    label="Descargar Ficha Go", 
                    data=result, 
                    file_name="FichaGo.xlsx", 
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == '__main__':
    main()