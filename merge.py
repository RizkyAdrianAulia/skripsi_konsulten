import argparse
import os
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Together

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
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")  # ✅ fixed typo here
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
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Menambahkan dataset baru: {len(new_chunks)}")
        new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunks_ids)
        db.persist()
    else:
        print("Tidak ada file dataset yang bisa dimasukkan")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

PROMPT_TEMPLATE = """
Jawablah pertanyaan berikut berdasarkan konteks yang diberikan. 
Kamu boleh menyimpulkan atau mengarang sedikit, selama masih konsisten dan tidak bertentangan dengan konteks.
Jangan menambahkan fakta yang tidak didukung oleh konteks.

Jika informasi tidak ditemukan dalam konteks, jawab dengan:
"Maaf, saya tidak menemukan informasi tersebut dalam konteks yang tersedia."

Konteks:
{context}

---

Pertanyaan: {question}

Jawaban:
"""

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)

    MIN_SCORE = 0.75
    filtered_results = [(doc, score) for doc, score in results if score >= MIN_SCORE]

    if not filtered_results:
        print("⚠️ Tidak ada konteks yang cukup relevan ditemukan. Tidak bisa menjawab.\n")
        return

    seen = set()
    unique_contents = []
    for doc, _ in filtered_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_contents.append(doc.page_content)

    context_text = "\n\n---\n\n".join(unique_contents)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.6,          
    max_tokens=1024,         
    top_p=0.85,               
    together_api_key="b4492aca173de7b5f27e5caba9af896acd0214a42cc8b9a2f75a93046562d4dd"
)

    response_text = model.invoke(prompt)
    final_answer = remove_repeated_sentences(response_text)

    sources = [doc.metadata.get("id", "No ID") for doc, _ in filtered_results]

    print("\n=== Jawaban ===")
    print(final_answer)
    print("\n=== Sumber ===")
    print(sources)
    print("\n")

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

    print("=== Konsultasi Pembuatan Konten ===")
    print("Ketik 'exit' untuk menyudahi pertanyaan.\n")

    while True:
        query_text = input("Masukkan pertanyaan kamu: ")
        if query_text.lower() in ["exit", "keluar", "quit"]:
            print("Terima kasih sudah menggunakan saya!")
            break
        query_rag(query_text)

if __name__ == "__main__":
    main()
