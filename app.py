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
import random

from io import BytesIO
from PIL import Image  
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
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
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor





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

contextualize_q_system_prompt = """Eres un programa que recibe licitaciones de diversas entidades.\
    Tu función es rellenar una ficha en base a preguntas especificas que se te harán de forma automátcia.\
    Si no encuentras la información que se te pide sigue buscando, debe estar en los documentos entregados."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_system_prompt = """Eres un programa que recibe licitaciones de diferentes entidades, deberas responder todas las preguntas que se te hagan en base estos documentos, \
    Al responder envia únicamente la respuesta, nunca respondas con 'La respuesta a tu pregunta es: ...', envia únicamente el contenido.
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


def extractPDF(pdf_docs):
    all_texts = []
    pages = []

    # Progress bar initialization
    my_bar1 = st.progress(0, text="Cargando documento")
    my_bar2 = st.progress(0, text="Convirtiendo documento a texto")

    # Load all pages from all PDF documents
    for n, pdf in enumerate(pdf_docs): 
        print(n)
        print(len(pdf_docs))
        my_bar1.progress((n+1) / len(pdf_docs), "Cargando documento")
        pdf_pages = convert_from_bytes(pdf.getvalue())
        pages.extend(pdf_pages)  # Append all pages to the pages list
    
    N_pages = len(pages)  # Total number of pages

    # Process each page
    for page_num, image in enumerate(pages):
        # Update progress bar
        my_bar2.progress((page_num + 1) / N_pages, "Convirtiendo documento a texto")

        # Extract text from the image using Tesseract
        txt = pytesseract.image_to_string(image, lang="spa")
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
        chunk_size=512,
        chunk_overlap=64,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_parent_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Define dimensions through an example embedding or directly through model assumption if documented.
    embedding_size = len(embeddings.embed_query("test text"))

    # Use the correct dimensionality when initializing FAISS
    index = faiss.IndexFlatL2(embedding_size)

    # The vectorstore to use to index the child chunks
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},  # Ensure proper handling of ids if they don't work as expected.
        relevance_score_fn=0.9
    )

    # Parent store (where you presumably keep the mapping of parent document ids)
    ret_store = InMemoryStore()

    # Create the retriever
    retriever = ParentDocumentRetriever(
        search_type="mmr",
        vectorstore=vectorstore,
        docstore=ret_store,
        search_kwargs={"k": 10},  # Adjust this value based on results.
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=32),
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=128),
    )

    return retriever
def get_contextual_retriever(retriever):
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever

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


def consistencia():
    print("TEST CONSISTENCIA")
    keys = {"Titulo":["B4","B4"], "Organismo":["B5","B5"], "Presu":["B6","B7"], "Sol_tec":["B12","B16"], "Equipo":["B15","B18"]}
    sheet1 = "A1 Resumen" # Objeto Worksheet de openpyxl
    sheet2 = "B1 Requisitos licitación"

    for _, value in keys.items():
        cell1 = st.session_state.tabla[sheet1][value[0]]  # Por ejemplo, 'B4'
        cell2 = st.session_state.tabla[sheet2][value[1]]  # Por ejemplo, 'B4'

        if cell1.value == "NO SE PUDO ENCONTRAR RESPUESTA" and cell2.value != "NO SE PUDO ENCONTRAR RESPUESTA":
            print("A MAL B BIEN")
            st.session_state.tabla.modify(sheet1, value[0], cell2.value)
        elif cell2.value == "NO SE PUDO ENCONTRAR RESPUESTA":
            print("B MAL A BIEN")
            st.session_state.tabla.modify(sheet2, value[1], cell1.value)
        else:
            print("AMBAS MAL")
            st.session_state.tabla.modify(sheet1, value[0], "Error de concistencia")
            st.session_state.tabla.modify(sheet2, value[1], "Error de concistencia")

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
    if "tabla" not in st.session_state:
        st.session_state.tabla = None


    # Configuracion pagina streamlit
    st.set_page_config(page_title="IA Chat", page_icon=Image.open('visual\\proyeco_logo.jpg'))
    st.write(css, unsafe_allow_html=True)
    st.header("INSTRUCCIONES")
    st.write("1.- Subir uno o varios archivos en formato PDF al recuadro ubicado a la izquierda de la página.")
    st.write("2.- Clicar en el Botón de procesar.")
    st.write("3.- Con el archivo ya procesado (puede tardar un par de minutos) se puede generar la ficha Go o realizar preguntas al chatbot con normalidad.")
    st.write("4.- De haber generado la ficha Go, clicar en el boton de 'Descargar ficha Go' para obtener el archivo excel.")
    st.write("5.- Cerrar el programa o continuar chateando con el bot.")
    st.write("⚠️Recordar que el uso de esta herramienta no es gratuito y su mal uso puede generar gastos imprevistos.")
    st.write("⚠️El programa sigue en estado de prueba, en caso de no seguir las instrucciones correctamente, reiniciar la página y repetir.")
    user_question = st.text_input(label="Texto", placeholder="Escribe aquí", key='widget', disabled = not st.session_state.button_clicked)

    with st.sidebar:
        st.subheader("Documentos")
        pdf_docs = st.file_uploader("Sube aquí tus archivos y presiona 'Procesar'", accept_multiple_files=True, type="pdf")

        if st.button("Procesar", disabled=st.session_state.button_clicked, use_container_width=True):
            if pdf_docs:
                st.session_state.button_clicked = True

                with st.status("Procesando", expanded=False, state="running"):
                    # Saca el texto del pdft
                    st.write("Extrayendo imagenes")
                    text_pages = extractPDF(pdf_docs)

                    processed_text = remove_similar_columns(" ".join(text_pages))

                    # Convierte el texto en chunks
                    text_chunks = get_text_chunks(processed_text)

                    docs = to_doc(text_chunks)

                    st.write("Creando retriever")
                    retriever = get_parent_retriever()

                    retriever.add_documents(docs, ids=None)

                    contextual_ret = get_contextual_retriever(retriever)

                    history_ret = get_history_aware_ret(contextual_ret)

                    st.write("Creando cadena")

                    rag_chain = create_retrieval_chain(history_ret, question_answer_chain)

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
            if st.session_state.button_clicked == True:
                st.session_state.button_generarFicha = True

                with open("Output.txt", "r+", encoding="utf-8") as text_file:


                    if st.session_state.button_clicked == True:
                        text_file.write("----------------------------------------------------------------------------------------------\n")
                        sheets = ["A1 Resumen", "B1 Requisitos licitación"]
                        st.session_state.tabla = tablaGo.tablaGo("docs\\ficha.xlsx","docs\\prompts.json", sheets)
                        aux = ""

                        with st.status("Procesando", expanded=False, state="running"):
                            
                            my_bar1 = st.progress(0, text="Página A1")
                            my_bar2 = st.progress(0, text="Página B1")

                            #Recorre el archivo json ubicando la lista A1 o B1
                            for i, system_questions in enumerate(st.session_state.tabla.questions):
                                keys_list = system_questions.keys()
                                casillas = st.session_state.tabla.casillas[i]
                                id = random.randint(1, 9999)
                                lista = [1]*len(keys_list)
                                n = 0

                                #Recorre cada apartado de las listas
                                while 1 in lista:
                                    res = ""

                                    #Obtiene iterativamente los elementos de la lista de los prompts
                                    #Si la n supera el tamaño de la lista de prompts del elemento en concreto, se escribe un false
                                    #En caso contrario, escribe el prompt n de la lista del diccionario
                                    prompts = st.session_state.tabla.next_q(i, n, lista)

                                    #Le pasa todos los prompts al llm y obtiene un array con sus respuestas
                                    response = st.session_state.conversational_rag_chain.batch(
                                        {"input":prompts },
                                        config={"configurable": {"session_id": id}},
                                    )["answer"]

                                    #Obtiene una lista de 0 y 1, 0 si obtuvo una respuesta correcta y 1 en caso contrario
                                    lista = st.session_state.tabla.contains_any_phrases_l(response, lista)
                                    print(lista)
                                    n += 1
                            

                        print("############################## Proceso terminado ##############################")
                        st.session_state.tabla.merge("Log", "A1", "Z180")
                        st.session_state.tabla.modify( "Log", "A1",aux) 

                        st.rerun()
                    
            else: st.warning("Por favor, procese algun documento antes de generar la ficha")
        if st.session_state.button_generarFicha:
            st.session_state.tabla.consistencia()
            st.download_button(
                label="Descargar Ficha Go", 
                use_container_width=True,
                data=st.session_state.tabla.save_file(),
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