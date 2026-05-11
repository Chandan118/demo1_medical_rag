__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import docx
import base64
import speech_recognition as sr
import subprocess
from streamlit_mic_recorder import mic_recorder
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f'''
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            '''
        st.markdown(md, unsafe_allow_html=True)

# --- UI Configuration ---
st.set_page_config(page_title="AI Medical Researcher", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🧬 AI Medical Researcher</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload Bio-Medical PDFs and Word Docs, and ask exact questions with citations.</p>', unsafe_allow_html=True)

# --- Configuration & Setup ---
DATA_DIR = "data"
DB_DIR = "chroma_db"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar for Data Upload and Processing ---
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader("Upload Medical Documents (Max 10)", type=["pdf", "docx"], accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if not uploaded_files and not os.listdir(DATA_DIR):
            st.warning("Please upload at least one PDF or place it in the data folder.")
        else:
            with st.spinner("Processing documents and generating embeddings..."):
                # Save uploaded files
                for uploaded_file in uploaded_files:
                    with open(os.path.join(DATA_DIR, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Read all documents
                raw_text = ""
                for filename in os.listdir(DATA_DIR):
                    file_path = os.path.join(DATA_DIR, filename)
                    if filename.endswith(".pdf"):
                        reader = PdfReader(file_path)
                        for page in reader.pages:
                            if page.extract_text():
                                raw_text += page.extract_text() + "\n"
                    elif filename.endswith(".docx"):
                        doc = docx.Document(file_path)
                        for para in doc.paragraphs:
                            if para.text:
                                raw_text += para.text + "\n"
                
                # Split text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(raw_text)
                
                # Create Embeddings and Vector DB
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=DB_DIR)
                
                # Setup LLM via Ollama
                try:
                    llm = Ollama(model="llama3")
                    # Test connection
                    llm.invoke("test")
                    
                    # Create Chain
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
                    )
                    st.success("System Ready! You can now chat with your documents.")
                except Exception as e:
                    st.error(f"Failed to connect to Ollama. Make sure 'ollama run llama3' is running in your terminal. Error: {e}")

    st.markdown("---")
    st.subheader("Tech Stack")
    st.markdown("- **LLM:** Llama-3 (Ollama)")
    st.markdown("- **Embeddings:** HuggingFace MiniLM")
    st.markdown("- **Vector DB:** ChromaDB")
    st.markdown("- **Framework:** LangChain")

# --- Main Chat Interface ---
st.subheader("2. Chat with Research Data")

col_chat, col_mic = st.columns([5, 1])
with col_mic:
    audio = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="🛑 Stop Recording", key='recorder')

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("E.g., What are the main challenges in robotic surgery?")

# Handle Audio Input
if audio:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio['bytes'])
    r = sr.Recognizer()
    with sr.AudioFile("temp_audio.wav") as source:
        audio_data = r.record(source)
        try:
            prompt = r.recognize_google(audio_data)
        except Exception:
            st.error("Could not understand audio. Please try speaking again.")

if prompt:
    if not st.session_state.qa_chain:
        st.error("Please process documents first using the sidebar!")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                try:
                    response = st.session_state.qa_chain.invoke(prompt)
                    answer = response['result']
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # --- VOICE MOOD: Text-to-Speech ---
                    subprocess.run(["edge-tts", "--text", answer, "--voice", "en-US-ChristopherNeural", "--write-media", "response.mp3"])
                    autoplay_audio("response.mp3")
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
