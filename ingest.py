import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

def ingest_pdfs(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    documents = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        clean_name = os.path.basename(uploaded_file.name)

        for d in docs:
            d.metadata["source"] = clean_name

        documents.extend(docs)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 400,
        separators = ['\n\n', '\n', ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore