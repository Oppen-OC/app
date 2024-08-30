import re
import os
import hmac
import json
import nltk
import pytesseract
import pandas as pd
import streamlit as st  
import tablaGo as tablaGo


from PIL import Image  
from io import BytesIO
from dotenv import load_dotenv
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
from pdfminer.high_level import extract_text
from visual.htmlTemplates import css, bot_template
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Explicitly set environment variables if needed
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

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


system_prompt = (
    "Eres un programa que recibe licitaciones de diferentes entidades, deberas responder todas las preguntas que se te hagan en base estos documentos, "
    "Todas las respuestas deben ser claras y concisas a menos de que se te diga lo contrario "
    "En caso de no saber la respuesta a una pregunta, responde con 'No lo se'"
    "NUNCA hagas referencia a la pregunta dentro del prompt, unicamente responde con la información que dispongas"
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

#Recibe al retriever y devuelve otro retrieevr que toma en cuenta el historial
def get_history(retriever):
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, prompt
    )

    return history_aware_retriever


# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    separators="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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
                for image in images:
                    ocr_text = pytesseract.image_to_string(image)
                    text += ocr_text
            except Exception as ocr_e:
                print(f"Error performing OCR: {ocr_e}")
                continue  # Skips if OCR fails

    if text:

        text = normalizar(text)

        # Write the extracted text to a file
        # with open("Output.txt", "w", encoding="utf-8") as text_file:
            # text_file.write(text)

    return text

def normalizar(txt):

    with open("docs\\stopwords.txt", 'r') as file: stopwords = file.read()

    txt = txt.lower()

    #Remueve puntuacion
    txt = re.sub(r'[^\w\s.,]', '', txt)

    #Tokeniza
    nltk.download('punkt')
    tokens = nltk.word_tokenize(txt, language='spanish')

    #Remueve stopwords
    tokens = [word for word in tokens if word not in stopwords]

    # Se devuelve todo a string
    processed_text = ' '.join(tokens)

    # Handling extra whitespaces (should not be necessary after join)
    processed_text = ' '.join(processed_text.split())
    
    return processed_text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print("SAVED")
    vectorstore.save_local("Otro")
    return vectorstore

def get_vectorstore_local():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    print("GOT LOCAL")
    return FAISS.load_local("Otro", embeddings, allow_dangerous_deserialization=True)


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

if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Streamlit main function
def main():

    load_dotenv()

    # Initialize session state variables
    if "button_clicked" not in st.session_state:
        st.session_state.button_clicked = False
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None


    # Configuracion pagina streamlit
    st.set_page_config(page_title="IA Chat", page_icon=Image.open('visual\\proyeco_logo.jpg'))
    st.write(css, unsafe_allow_html=True)
    st.header("Pregunta sobre tu PDF")
    user_question = st.text_input(label="Texto", placeholder="Escribe aquí", key='widget')

    # Procesa input
    if user_question:
        if st.session_state.button_clicked:
            rag_chain = st.session_state.rag_chain  # Get rag_chain from session state
            if rag_chain is not None:
                #response = rag_chain.invoke({"input": get_prompt(str(user_question)), 'chat_history': st.session_state.chat_history})
                response = rag_chain.invoke({"input": str(user_question),'context': get_history(st.session_state.retriever), 'chat_history': st.session_state.chat_history})
                # print(response)

                st.session_state.chat_history.extend([HumanMessage(content=user_question), response["answer"]])
                
                st.write(bot_template.replace("{{MSG}}", response["answer"]), unsafe_allow_html=True)
            else:
                st.warning("El rag_chain no está configurado correctamente. Por favor, procese los documentos nuevamente.")
        else:
            st.warning("Por favor, suba y procese los documentos")

    with st.sidebar:
        st.subheader("Documentos")
        pdf_docs = st.file_uploader("Sube aquí tus archivos y presiona 'Procesar'", accept_multiple_files=True, type="pdf")

        if st.button("Procesar"):
            if pdf_docs:
                st.session_state.button_clicked = True

                with st.spinner("Procesando"):
                    # Saca el texto del pdff
                    raw_text = get_pdf_text(pdf_docs)

                    # Convierte el texto en chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Consigue el vector de la base de datos
                    # vectorstore = get_vectorstore(text_chunks)
                    vectorstore = get_vectorstore_local()

                    # Convierte al vector store en un retriever, esto facilita la recuperacion de datos
                    retriever = vectorstore.as_retriever(k = 7)
                    st.session_state.retriever = retriever
                    print(retriever)

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
                                print(f"{key} | {question} | {response['answer']}")

                                if tabla.contains_any_phrases(response['answer'], tabla.err):
                                    print("necesitamos otra")

                                else:
                                    res = response['answer'] 

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


    
if __name__ == "__main__":
    main()
