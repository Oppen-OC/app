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
from tablaGo import tablaGo
import json

embeddings = None
pdf_docs = None

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
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print(vectorstore.index.ntotal)
    vectorstore.save_local("faiss_index")
    return vectorstore

def get_vectorstore_local():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def get_conversation_chain(vectorstore):
    
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
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    memory.chat_memory.add_ai_message(user_instruction)
    memory.chat_memory.add_ai_message(system_message)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        if st.session_state.chat_history and len(st.session_state.chat_history) > 2:
            st.error("Conversation object is None. Cannot handle user input.")
        return
        
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
            
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
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Documentos")
        pdf_docs = st.file_uploader("Sube aquí tus archivos y presiona 'Procesar'", accept_multiple_files=True, type  = "pdf")

        if pdf_docs:
            st.session_state.file_uploaded = True
            print("TRUE PDF")
        else:
            st.session_state.file_uploaded = False
            print("FALSE PDF")

        if st.button("Procesar"):
            if st.session_state.file_uploaded:
                st.session_state.button_clicked = True
                print("Procesar presionado")
                #st.rerun() #Sin esto no se actualiza el estado
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = get_vectorstore_local()
                    st.session_state.conversation = get_conversation_chain(st.session_state.vector_store)
                    print("GOT THE VECTOR STORE")
                
                st.rerun()

            else:
                st.warning("Por favor suba un documento antes de procesar")
                print("FILE UPLOADED IS FALSE")


        if st.session_state.file_uploaded and st.session_state.button_clicked:
            with st.spinner("Procesando"):
                #raw_text = get_pdf_text(pdf_docs)
                #text_chunks = get_text_chunks(raw_text)
                #st.session_state.vector_store = get_vectorstore(text_chunks)
                print(st.session_state.conversation)

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