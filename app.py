try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

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

# --- Page Config ---
st.set_page_config(
    page_title="MedLens AI | Clinical Research Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark premium background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1425 50%, #0a1020 100%);
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1525 0%, #111d35 100%);
    border-right: 1px solid rgba(99, 179, 237, 0.15);
}

/* Premium hero header */
.hero-container {
    background: linear-gradient(135deg, rgba(14,165,233,0.1) 0%, rgba(99,102,241,0.1) 50%, rgba(168,85,247,0.1) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(99,179,237,0.05) 0%, transparent 60%);
    animation: rotate 20s linear infinite;
}

@keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #63b3ed, #9f7aea, #ed64a6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: rgba(226,232,240,0.6);
    margin-top: 8px;
    font-weight: 400;
    letter-spacing: 0.5px;
}

.badge {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* Status cards */
.status-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.status-ready {
    border-color: rgba(72,187,120,0.4);
    background: rgba(72,187,120,0.05);
}

.status-idle {
    border-color: rgba(99,179,237,0.3);
    background: rgba(99,179,237,0.05);
}

/* Chat bubbles */
.chat-user {
    background: linear-gradient(135deg, #1e3a5f, #1a3352);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 18px 18px 4px 18px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #e2e8f0;
}

.chat-ai {
    background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(168,85,247,0.1));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #e2e8f0;
}

/* Sidebar labels */
.sidebar-label {
    color: rgba(148,163,184,0.8);
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}

/* Tech stack pills */
.tech-pill {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 500;
    margin: 2px;
}

/* Metrics */
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #63b3ed;
}

.metric-label {
    font-size: 0.75rem;
    color: rgba(148,163,184,0.7);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #63b3ed !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(99,179,237,0.3) !important;
    border-radius: 12px !important;
    background: rgba(99,179,237,0.03) !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(99,179,237,0.25) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.3), transparent);
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.markdown(md, unsafe_allow_html=True)

# --- Hero Header ---
st.markdown("""
<div class="hero-container">
    <div class="badge">🔬 Clinical AI Platform · Enterprise Grade</div>
    <p class="hero-title">MedLens AI</p>
    <p class="hero-subtitle">Conversational Intelligence for Biomedical Research · Powered by LLaMA-3 · RAG Architecture</p>
</div>
""", unsafe_allow_html=True)

# --- Config ---
DATA_DIR = "data"
DB_DIR = "chroma_db"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

# --- Sidebar ---
with st.sidebar:
    st.markdown('<p class="hero-title" style="font-size:1.4rem; margin-bottom:4px;">⚙️ Control Panel</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-label">📂 Document Ingestion</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDF or DOCX files here",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡ Process & Build Knowledge Base", use_container_width=True):
        if not uploaded_files and not os.listdir(DATA_DIR):
            st.warning("Upload at least one document first.")
        else:
            with st.spinner("🔄 Processing documents..."):
                for uf in uploaded_files:
                    with open(os.path.join(DATA_DIR, uf.name), "wb") as f:
                        f.write(uf.getbuffer())
                raw_text = ""
                doc_count = 0
                for filename in os.listdir(DATA_DIR):
                    fp = os.path.join(DATA_DIR, filename)
                    if filename.endswith(".pdf"):
                        reader = PdfReader(fp)
                        for page in reader.pages:
                            if page.extract_text():
                                raw_text += page.extract_text() + "\n"
                        doc_count += 1
                    elif filename.endswith(".docx"):
                        doc = docx.Document(fp)
                        for para in doc.paragraphs:
                            if para.text:
                                raw_text += para.text + "\n"
                        doc_count += 1

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
                chunks = text_splitter.split_text(raw_text)
                st.session_state.doc_count = doc_count

                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=DB_DIR)

                try:
                    llm = Ollama(model="llama3")
                    llm.invoke("test")
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
                    )
                    st.success(f"✅ {doc_count} docs | {len(chunks)} chunks indexed!")
                except Exception as e:
                    st.error(f"Ollama connection failed. Run `ollama run llama3` in terminal. Error: {e}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Status indicator
    if st.session_state.qa_chain:
        st.markdown("""
        <div class="status-card status-ready">
            <span style="color:#48bb78; font-weight:700;">● SYSTEM READY</span><br>
            <span style="color:rgba(148,163,184,0.7); font-size:0.8rem;">Knowledge base active</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card status-idle">
            <span style="color:#63b3ed; font-weight:700;">○ AWAITING DOCUMENTS</span><br>
            <span style="color:rgba(148,163,184,0.7); font-size:0.8rem;">Upload & process to begin</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">🛠 Technology Stack</p>', unsafe_allow_html=True)
    st.markdown("""
    <span class="tech-pill">LLaMA-3</span>
    <span class="tech-pill">LangChain</span>
    <span class="tech-pill">ChromaDB</span>
    <span class="tech-pill">HuggingFace</span>
    <span class="tech-pill">RAG</span>
    <span class="tech-pill">Edge-TTS</span>
    """, unsafe_allow_html=True)

# --- Stats Row ---
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{st.session_state.doc_count}</div>
        <div class="metric-label">Documents Loaded</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(st.session_state.messages)}</div>
        <div class="metric-label">Queries Asked</div>
    </div>""", unsafe_allow_html=True)
with col3:
    status = "Active" if st.session_state.qa_chain else "Idle"
    color = "#48bb78" if st.session_state.qa_chain else "#63b3ed"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color};">{status}</div>
        <div class="metric-label">AI Engine Status</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --- Chat Interface ---
st.markdown("### 💬 Research Query Interface")

col_chat, col_mic = st.columns([6, 1])
with col_mic:
    st.markdown("<br>", unsafe_allow_html=True)
    audio = mic_recorder(start_prompt="🎙️", stop_prompt="🛑", key='recorder')

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask anything about your medical documents...")

if audio:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio['bytes'])
    r = sr.Recognizer()
    with sr.AudioFile("temp_audio.wav") as source:
        audio_data = r.record(source)
        try:
            prompt = r.recognize_google(audio_data)
        except Exception:
            st.error("Could not understand audio. Please try again.")

if prompt:
    if not st.session_state.qa_chain:
        st.error("⚠️ Please upload documents and click 'Process & Build Knowledge Base' first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔬 Analyzing research corpus..."):
                try:
                    response = st.session_state.qa_chain.invoke(prompt)
                    answer = response['result']
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    subprocess.run(["edge-tts", "--text", answer[:500], "--voice", "en-US-ChristopherNeural", "--write-media", "response.mp3"])
                    autoplay_audio("response.mp3")
                except Exception as e:
                    st.error(f"Error: {e}")
