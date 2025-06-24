import os
import shutil
import re
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Together
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from streamlit_chat import message

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(DATA_PATH, filename))
            documents.extend(loader.load())
    return documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    return chunks

def add_to_chroma(chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    if new_chunks:
        new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunks_ids)
        db.persist()

# PROMPT_TEMPLATE = """
# ### Instruksi:
# Anda adalah asisten profesional yang menjawab dengan ringkas, langsung ke inti, dan hanya dalam Bahasa Indonesia. 
# Jawab **hanya** berdasarkan konteks. **Jangan menambahkan pertanyaan lain**, jangan menebak, jangan membuat revisi, dan jangan mengulang jawaban.

# Jika tidak ada informasi dalam konteks, jawab dengan:
# "Maaf, saya tidak menemukan informasi tersebut dalam konteks yang tersedia."

# ### SYSTEM: Anda hanya menjawab satu pertanyaan berdasarkan konteks. Jangan tanya balik. Jangan lanjutkan topik lain. Jangan tulis "pertanyaan" atau "revisi".

# ### Konteks:
# {context}

# ### Pertanyaan:
# {question}

# ### Jawaban:
# """

PROMPT_TEMPLATE = """
### INSTRUKSI (WAJIB DIIKUTI):

- Jawablah HANYA dalam Bahasa Indonesia.
- Jangan ulangi kalimat.
- Jangan membuat pertanyaan tambahan.
- Jangan menggunakan Bahasa Inggris dalam kondisi apapun.
- Jangan menjawab jika informasi tidak ditemukan dalam konteks.

Jika tidak ada informasi dalam konteks, cukup jawab:
"Maaf, saya tidak menemukan informasi tersebut dalam konteks yang tersedia."

### KONTEKS:
{context}

### PERTANYAAN:
{question}

### JAWABAN:
"""





def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=8)
    MIN_SCORE = 0.6
    filtered_results = [(doc, score) for doc, score in results if score >= MIN_SCORE]

    if not filtered_results:
        return "Tidak ada konteks yang cukup relevan ditemukan. Tidak bisa menjawab."

    seen = set()
    unique_contents = []
    for doc, _ in filtered_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_contents.append(doc.page_content)

    context_text = "\n\n---\n\n".join(unique_contents)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, question=query_text
    )

    # model = Together(
    #     model="mistralai/Mistral-7B-Instruct-v0.2",
    #     temperature=0.5,
    #     max_tokens=512,
    #     top_p= 0.85,
    #     together_api_key="b4492aca173de7b5f27e5caba9af896acd0214a42cc8b9a2f75a93046562d4dd"
    # )


    model = Together(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        temperature=0.5,
        max_tokens=512,
        top_p= 0.85,
        together_api_key="b4492aca173de7b5f27e5caba9af896acd0214a42cc8b9a2f75a93046562d4dd"
    )    

    # response_text = model.invoke(prompt)
    response_text = model.invoke(prompt, stop=["\n###", "Pertanyaan:", "Revisi:", "Question:"])
    return remove_repeated_sentences(response_text)

#Untuk mereduksi kata yang berulang
def remove_repeated_sentences(text):
    sentences = text.split(". ")
    seen = set()
    unique_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)
    return ". ".join(unique_sentences).strip()

# Coba buat reduksi bahasa inggris
def clean_response(text):
    sentences = text.split(". ")
    seen = set()
    result = []
    for s in sentences:
        s_clean = s.strip().lower()
        if s_clean and s_clean not in seen and not re.search(r"[a-z]{3,}", s) if "dan" not in s.lower() else False:  # filter Inggris
            seen.add(s_clean)
            result.append(s.strip())
    return ". ".join(result).strip()

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

# UI Streamlit
st.set_page_config(page_title="Konsulten", page_icon="💬")
st.title("Konsultasi Pembuatan Konten")

if "messages" not in st.session_state:
    st.session_state.messages = []

# with st.sidebar:
#     if st.button("🔄 Reset Database & Load PDF"):
#         clear_database()
#         docs = load_documents()
#         chunks = split_documents(docs)
#         add_to_chroma(chunks)
#         st.success("📚 Data berhasil dimuat ulang!")

prompt = st.chat_input("Silahkan tanya sesuatu...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Menjawab..."):
        response = query_rag(prompt)
    st.session_state.messages.append({"role": "ai", "content": response})

for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=(msg["role"] == "user"), key=f"msg-{i}")



