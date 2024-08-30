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
from pdf2image import convert_from_bytes
from pdfminer.high_level import extract_text
from visual.htmlTemplates import css, bot_template
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
from sentence_transformers import CrossEncoder
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chains import create_extraction_chain
from typing import Optional, List
from langchain.chains import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel
from langchain import hub

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

contextualize_q_system_prompt = """Eres un programa que recibe licitaciones de entidades estatales\
    Tu función es rellenar ujna ficha en base a preguntas especificas que se te harán de forma automátcia\
    Si no encuentras la informaciñon que se te pide sigue buscando, debe estar en los documentos entregados."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

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

store = {}

obj = hub.pull("wfh/proposal-indexing")
llm = ChatOpenAI(model='gpt-3.5-turbo')
runnable = obj | llm

class Sentences(BaseModel):
    sentences: List[str]
    
# Extraction
extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=llm)
def get_propositions(text):
    runnable_output = runnable.invoke({
    	"input": text
    }).content
    propositions = extraction_chain.invoke(runnable_output)["text"][0].sentences
    return propositions
    
paragraphs = text.split("\n\n")
text_propositions = []
for i, para in enumerate(paragraphs[:5]):
    propositions = get_propositions(para)
    text_propositions.extend(propositions)
    print (f"Done with {i}")

print (f"You have {len(text_propositions)} propositions")
print(text_propositions[:10])

print("#### Agentic Chunking ####")

from agentic_chunker import AgenticChunker
ac = AgenticChunker()
ac.add_propositions(text_propositions)
print(ac.pretty_print_chunks())
chunks = ac.get_chunks(get_type='list_of_strings')
print(chunks)
documents = [Document(page_content=chunk, metadata={"source": "local"}) for chunk in chunks]
rag(documents, "agentic-chunks")

def get_history_aware_ret(retriever):
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Function to format documents
def format_docs(docs: Document) -> str:
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

    return text

import streamlit as st
from pdf2image import convert_from_bytes
import pytesseract

def protectExtractPDF(pdf_docs):
    all_texts = []
    pages = []

    # Progress bar initialization
    my_bar = st.progress(0, text="Convirtiendo a texto documento")

    # Load all pages from all PDF documents
    for pdf in pdf_docs:
        pdf_pages = convert_from_bytes(pdf.getvalue())
        pages.extend(pdf_pages)  # Append all pages to the pages list
    
    N_pages = len(pages)  # Total number of pages

    # Process each page
    for page_num, image in enumerate(pages):
        # Update progress bar
        my_bar.progress((page_num + 1) / N_pages, "Convirtiendo a texto documento")

        # Extract text from the image using Tesseract
        txt = pytesseract.image_to_string(image)
        print(f"Página {page_num + 1} procesada.")
        
        # Accumulate the extracted text for saving later
        all_texts.append(f"Page {page_num + 1}:\n{txt}\n")
    
    # Write all extracted text to a file
    with open("textoContexto.txt", 'w', encoding='utf-8') as file:
        file.write("\n".join(all_texts))

    return all_texts


def to_doc(text_chunks: list[str],  metadata=None) -> Document:
    documents = []
    for i, chunk in enumerate(text_chunks):
        if isinstance(metadata, list):
            doc_metadata = metadata[i] if i < len(metadata) else {}
        else:
            doc_metadata = metadata or {}
        documents.append(Document(page_content=chunk, metadata=doc_metadata))
    return documents

def remove_similar_columns(text):
    # Separar líneas del texto
    lines = text.splitlines()
    
    # Filtrar líneas que son sospechosas de ser columnas repetitivas
    filtered_lines = []
    for line in lines:
        # Eliminar espacios en blanco al inicio y al final
        clean_line = line.strip()
        
        # Ignorar líneas muy cortas o que consisten en una palabra repetitiva
        if len(clean_line) > 1 and not re.fullmatch(r'(\w)\1*', clean_line):
            filtered_lines.append(clean_line)
    
    # Unir las líneas filtradas
    filtered_text = '\n'.join(filtered_lines)
    
    return filtered_text

def normalizar(txt_list: list) -> str:

    with open("docs\\stopwords.txt", 'r') as file:
        stopwords = file.read().splitlines()  # Asegúrate de dividir las líneas

    nltk.download('punkt')

    processed_texts = []

    for txt in txt_list:
        txt = unicodedata.normalize('NFD', txt)
        # Filtrar los caracteres que no son acentos diacríticos
        txt = ''.join(c for c in txt if unicodedata.category(c) != 'Mn')

        # Minusculas
        txt = txt.lower()

        # Remueve puntuacion, excepto guiones que están solos o seguidos de un solo dígito
        txt = re.sub(r'(?<!\d)-|\b\w-(?!\d)\b|[^\w\s.,-]', '', txt)

        # Tokeniza
        tokens = nltk.word_tokenize(txt, language='spanish')

        # Remueve stopwords
        tokens = [word for word in tokens if word not in stopwords]

        # Se devuelve todo a string
        processed_text = ' '.join(tokens)

        # Manejo de espacios extra (normalmente innecesario después de join)
        processed_text = ' '.join(processed_text.split())

        # Append processed text to the list
        processed_texts.append(processed_text)

    # Join all processed texts into a single string
    final_text = ' '.join(processed_texts)

    return final_text


def get_text_chunks(text: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # This text splitter is used to create the parent documents
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=100)

    # This text splitter is used to create the child documents, it should create documents smaller than the parent
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=51)

    # The vectorstore to use to index the child chunks
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=faiss.IndexFlatL2(len(embeddings.embed_query("Maduro mamaguevo"))),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # The storage layer for the parent documents
    ret_store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=ret_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    return retriever

#def get_vectorstore_local() -> FAISS:
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    #print("GOT LOCAL")
    #return Chroma.load_local("Otro", embeddings, allow_dangerous_deserialization=True)

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
    if 'button_generarFicha' not in st.session_state:
        st.session_state.button_generarFicha = False
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
        rag_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "conversational_rag_chain" not in st.session_state:
        st.session_state.conversational_rag_chain = None

    # Configuracion pagina streamlit
    st.set_page_config(page_title="IA Chat", page_icon=Image.open('visual\\proyeco_logo.jpg'))
    st.write(css, unsafe_allow_html=True)
    st.header("Pregunta sobre tu PDF")
    user_question = st.text_input(label="Texto", placeholder="Escribe aquí", key='widget')

    with st.sidebar:
        st.subheader("Documentos")
        pdf_docs = st.file_uploader("Sube aquí tus archivos y presiona 'Procesar'", accept_multiple_files=True, type="pdf")

        if st.button("Procesar", disabled=st.session_state.button_clicked, use_container_width=True):
            if pdf_docs:
                st.session_state.button_clicked = True

                with st.status("Procesando", expanded=False, state="running"):
                    # Saca el texto del pdf
                    # raw_text = get_pdf_text(pdf_docs)
                    # extract_text_with_context(pdf_docs)
                    st.write("Extrayendo imagenes")
                    text_pages = protectExtractPDF(pdf_docs)

                    processed_text = remove_similar_columns(" ".join(text_pages))

                    #with open("texto_norm1.txt", 'w', encoding='utf-8') as file:
                        #file.write(processed_text)

                    #st.write("Normalizando texto")
                    #processed_text = normalizar(text_pages)
                    
                    #with open("texto_norm2.txt", 'w', encoding='utf-8') as file:
                        #file.write(processed_text)

                    # Convierte el texto en chunks
                    text_chunks = get_text_chunks(processed_text)

                    docs = to_doc(text_chunks)

                    st.write("Creando retriever")
                    retriever = get_retriever()

                    retriever.add_documents(docs, ids=None)

                    history_ret = get_history_aware_ret(retriever)

                    rag_chain = create_retrieval_chain(history_ret, question_answer_chain)

                    st.write("Creando cadena")
                    conversational_rag_chain = RunnableWithMessageHistory(
                        rag_chain,
                        get_session_history,
                        input_messages_key="input",
                        history_messages_key="chat_history",
                        output_messages_key="answer",
                    )

                    st.session_state.conversational_rag_chain = conversational_rag_chain

                    print("Final del proceso")

                    st.rerun()

            else:
                st.warning("Por favor, suba un documento antes de procesar")

        if st.button("Generar ficha go",disabled=st.session_state.button_generarFicha, use_container_width=True):
            st.session_state.button_generarFicha = True

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
                                response = st.session_state.conversational_rag_chain.invoke(
                                    {"input":question },
                                    config={"configurable": {"session_id": "123"}},
                                )["answer"]

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
                    use_container_width=True,
                    data=tabla.save_file(), 
                    file_name="result_FichaGo.xlsx", 
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