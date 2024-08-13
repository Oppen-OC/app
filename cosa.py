import os
from dotenv import load_dotenv  # Assuming dotenv for environment variables
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
import streamlit as st  # Assuming streamlit for the web interface
from PIL import Image  # For loading the image
import bs4  # This is imported but not used in the script
from xtx.htmlTemplates import css, bot_template, user_template
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pdfminer.high_level import extract_text
from pdf2image import convert_from_bytes
from io import BytesIO
import pytesseract


# Load environment variables
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini")

system_prompt = (
    "Eres un programa que recibe licitaciones del estado y responde preguntas en base a ellas. "
    "Si se se dice que respondas unicamente con si o no una pregunta, respondes unicamente con si o no "
    "Si no conoces la respuesta a una preguta tienes que decirlo claramente "
    "Todas las respuestas deben ser claras y concisas a menos de que se te diga lo contrario "
    "Todas las respuestas tienen que estar basadas en el documento de licitacion"
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

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
        # Write the extracted text to a file
        with open("Output.txt", "w", encoding="utf-8") as text_file:
            text_file.write(text)


    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("ADIF")
    return vectorstore

def get_vectorstore_local():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.load_local("ADIF", embeddings, allow_dangerous_deserialization=True)


def get_prompt(input):
    prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ]
                )
    return prompt

# Streamlit main function
def main():

    load_dotenv()

    # Initialize session state variables
    if "button_clicked" not in st.session_state:
        st.session_state.button_clicked = False
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    # Configuracion pagina streamlit
    st.set_page_config(page_title="IA Chat", page_icon=Image.open('img/proyeco_logo.jpg'))
    st.write(css, unsafe_allow_html=True)
    st.header("Pregunta sobre tu PDF")
    user_question = st.text_input(label="Texto", placeholder="Escribe aquí", key='widget')

    # Procesa input
    if user_question:
        if st.session_state.button_clicked:
            rag_chain = st.session_state.rag_chain  # Get rag_chain from session state
            if rag_chain is not None:
                response = rag_chain.invoke({"input": str(user_question)})
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
                    # Saca el texto del pdf
                    raw_text = get_pdf_text(pdf_docs)

                    # Convierte el texto en chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Consigue el vector de la base de datos
                    vectorstore = get_vectorstore(text_chunks)

                    # Convierte al vector store en un retriever, esto facilita la recuperacion de datos
                    retriever = vectorstore.as_retriever()

                    # Especifica el contexto antes de pasarselo al llm
                    question_answer_chain = create_stuff_documents_chain(llm, prompt)

                    # Propaga el contexto por toda la cadena
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)  
                    st.session_state.rag_chain = rag_chain

                    print("Final del proceso")

                    st.rerun()
            else:
                st.warning("Por favor suba un documento antes de procesar")

if __name__ == "__main__":
    main()