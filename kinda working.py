import re
import os
import hmac
import json
import nltk
import pytesseract
import pandas as pd
import streamlit as st  
import tablaGo as tablaGo
import unicodedata
import faiss

from PIL import Image  
from io import BytesIO
from dotenv import load_dotenv
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
from pdfminer.high_level import extract_text
from visual.htmlTemplates import css, bot_template
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

# Load environment variables
load_dotenv()

# Explicitly set environment variables if needed
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"

with open('docs\\config.json') as f:
    config = json.load(f)

# Inicializa el LLM 
model_params = config['model']['parameters']

llm = ChatOpenAI(
    model_name=config['model']['version'],
    temperature=model_params['temperature'],
    max_tokens=model_params['max_tokens'],
    top_p=model_params['top_p'],
    frequency_penalty=model_params['frequency_penalty'],
    presence_penalty=model_params['presence_penalty'],
)

contextualize_q_system_prompt = """Dada una historia de chat y la última pregunta del usuario \
que podría hacer referencia al contexto en la historia del chat, formula una pregunta independiente \
que pueda entenderse sin la historia del chat. NO respondas la pregunta, \
solo reformúlala si es necesario y, de lo contrario, devuélvela tal como está."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

def get_history_aware_ret(retriever):
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

qa_system_prompt = """Eres un programa que recibe licitaciones de diferentes entidades, deberas responder todas las preguntas que se te hagan en base estos documentos, \


    {context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_pdf_text(pdf_docs) -> str:
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
                for image in images:
                    ocr_text = pytesseract.image_to_string(image)
                    text += ocr_text
            except Exception as ocr_e:
                print(f"Error performing OCR: {ocr_e}")
                continue  # Skips if OCR fails

    with open("texto.txt", 'w', encoding='utf-8') as file:
        file.write(text)

    text = normalizar(text)

    with open("texto_norm.txt", 'w', encoding='utf-8') as file:
        file.write(text)

    return text

def to_doc(text_chunks,  metadata=None):
    documents = []
    for i, chunk in enumerate(text_chunks):
        if isinstance(metadata, list):
            doc_metadata = metadata[i] if i < len(metadata) else {}
        else:
            doc_metadata = metadata or {}
        documents.append(Document(page_content=chunk, metadata=doc_metadata))
    return documents

def normalizar(txt: str) -> str:

    with open("docs\\stopwords.txt", 'r') as file:
        stopwords = file.read().splitlines()  # Asegúrate de dividir las líneas

    txt = unicodedata.normalize('NFD', txt)
    # Filtrar los caracteres que no son acentos diacríticos
    txt = ''.join(c for c in txt if unicodedata.category(c) != 'Mn')

    # Minusculas
    txt = txt.lower()

    # Remueve puntuacion, excepto guiones que están solos o seguidos de un solo dígito
    txt = re.sub(r'(?<!\d)-|\b\w-(?!\d)\b|[^\w\s.,-]', '', txt)

    # Remueve secuencias de puntos (ejemplo "...")
    patron = r'\.{3,}'
    txt = re.sub(patron, '', txt)

    # Tokeniza
    nltk.download('punkt')
    tokens = nltk.word_tokenize(txt, language='spanish')

    # Remueve stopwords
    tokens = [word for word in tokens if word not in stopwords]

    # Se devuelve todo a string
    processed_text = ' '.join(tokens)

    # Manejo de espacios extra (normalmente innecesario después de join)
    processed_text = ' '.join(processed_text.split())
    
    return processed_text


def get_text_chunks(text: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks: list[str]) -> FAISS:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print("SAVED")
    vectorstore.save_local("Otro")
    return vectorstore

def get_emebddings(txt_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    doc_embeddings = [embeddings.embed_text(doc) for doc in txt_chunks]
    dimension = len(doc_embeddings[0])  # Dimension of your embeddings
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search

    # Convert embeddings to numpy array and add them to the index
    faiss_index = FAISS(index, doc_embeddings)

    # Add documents to the FAISS vector store
    vector_store = FAISS.from_documents(documents, embeddings)


def get_retriever():
    # This text splitter is used to create the parent documents
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
    # This text splitter is used to create the child documents
    # It should create documents smaller than the parent
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)
    # The vectorstore to use to index the child chunks
    vectorstore = FAISS(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    index=faiss.IndexFlatL2(len(OpenAIEmbeddings(model="text-embedding-3-large").embed_query("hello world"))),
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
    # The storage layer for the parent documents
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    return retriever

# Initialize the retriever
#retriever = ParentDocumentRetriever(
    #vectorstore=vectorstore,
    #docstore=store,
    #child_splitter=child_splitter,
    #parent_splitter=parent_splitter,
#)


def get_vectorstore_local() -> FAISS:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    print("GOT LOCAL")
    return Chroma.load_local("Otro", embeddings, allow_dangerous_deserialization=True)

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("Contraseña incorrecta")
    return False

#if not check_password():
    #st.stop()  # Do not continue if check_password is not True.

# Streamlit main function
def main():

    load_dotenv()

    # Initialize session state variables
    if "button_clicked" not in st.session_state:
        st.session_state.button_clicked = False
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
        rag_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "conversational_rag_chain" not in st.session_state:
        st.session_state.conversational_rag_chain = None
    if "store" not in st.session_state:
        st.session_state.store = InMemoryStore()



    # Configuracion pagina streamlit
    st.set_page_config(page_title="IA Chat", page_icon=Image.open('visual\\proyeco_logo.jpg'))
    st.write(css, unsafe_allow_html=True)
    st.header("Pregunta sobre tu PDF")
    user_question = st.text_input(label="Texto", placeholder="Escribe aquí", key='widget')

    with st.sidebar:
        st.subheader("Documentos")
        pdf_docs = st.file_uploader("Sube aquí tus archivos y presiona 'Procesar'", accept_multiple_files=True, type="pdf")

        if st.button("Procesar"):
            if pdf_docs:
                st.session_state.button_clicked = True

                with st.spinner("Procesando"):
                    # Saca el texto del pdf
                    raw_text = get_pdf_text(pdf_docs)

                    # Convierte el texto en chunks
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.text_chunks = text_chunks

                    docs = to_doc(text_chunks)

                    # Consigue el vector de la base de datos
                    # vectorstore = get_vectorstore(text_chunks)
                    # st.session_state.vectorstore = vectorstore
                    # print(st.session_state.vectorstore)

                    # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
                    # retriever = ParentDocumentRetriever(
                        #vectorstore = FAISS.from_documents(docs, embedding = OpenAIEmbeddings(model="text-embedding-3-large")),
                        #docstore=InMemoryStore(),
                        #child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
                    #)

                    retriever = get_retriever()

                    retriever.add_documents(docs, ids=None)
                    print(retriever)

                    history_ret = get_history_aware_ret(retriever)

                    rag_chain = create_retrieval_chain(history_ret, question_answer_chain)

                    conversational_rag_chain = RunnableWithMessageHistory(
                        rag_chain,
                        get_session_history,
                        input_messages_key="input",
                        history_messages_key="chat_history",
                        output_messages_key="answer",
                    )
                    st.session_state.conversational_rag_chain = conversational_rag_chain

                    st.session_state.rag_chain = rag_chain

                    print("Final del proceso")

                    st.rerun()

            else:
                st.warning("Por favor, suba un documento antes de procesar")

        if st.button("Generar ficha go"):
            with st.spinner("Procesando"):

                with open("Output.txt", "w", encoding="utf-8") as text_file:
                    
                    df = None
                    tabla = None

                    if st.session_state.button_clicked == True:
                        text_file.write("----------------------------------------------------------------------------------------------\n")
                        tabla = tablaGo.tablaGo("docs\\ficha.xlsx","docs\\prompts.json")
                        system_questions =  tabla.questions
                        casillas = tabla.casillas
                        print(system_questions)
                        keys_list = list(system_questions.keys())
                        cont = 0
                        df = pd.read_excel('docs\\ficha.xlsx', sheet_name='B1 Requisitos licitación')

                        for key in keys_list:
                            res = ""

                            for question in system_questions[key]:
                                response = st.session_state.rag_chain.invoke({"input": str(question),'context': get_history(st.session_state.retriever), 'chat_history': st.session_state.chat_history})
                                print(f"{key} | {question} | {response}")

                                if tabla.contains_any_phrases(response, tabla.err):
                                    print("necesitamos otra")

                                else:
                                    res = response

                                    # Generar archivo .txt
                                    text_file.write(f"{key} | {question} | {res}\n")
                                    text_file.write("----------------------------------------------------------------------------------------------\n")

                                    # Generar archivo xlsx
                                    tabla.modify("B1 Requisitos licitación", casillas[cont], res)
                                    cont += 1

                                    break

                            if res == "":
                                text_file.write(f"{key} | {question} | {"NO SE PUDO ENCONTRAR RESPUESTA"}\n")
                                text_file.write("----------------------------------------------------------------------------------------------\n")
                                tabla.modify("B1 Requisitos licitación", casillas[cont], "NO SE PUDO ENCONTRAR RESPUESTA") 
                                cont += 1
                        print("############################## Proceso terminado ##############################")
                    
                    else:
                        st.warning("Por favor, procese algun documento antes de genesrar la ficha")

            if df.bool is not None and tabla is not None:
                st.download_button(
                    label="Descargar Ficha Go", 
                    data=tabla.save_file(), 
                    file_name="FichaGo.xlsx", 
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    # Procesa input
    if user_question:
        if st.session_state.button_clicked:

            response = st.session_state.conversational_rag_chain.invoke(
                {"input":user_question },
                config={"configurable": {"session_id": "123"}},
            )["answer"]

            st.session_state.chat_history.extend([HumanMessage(content=user_question), response])
            #print(st.session_state.chat_history)

            st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)

        else:
            st.warning("Por favor, suba y procese los documentos")

    
if __name__ == "__main__":
    main()
