import os
import faiss
import pickle
import numpy as np
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embedding_model = OpenAIEmbeddings()

def create_vector_store(docs, save_path="vectorstore/jd"):
    os.makedirs(save_path, exist_ok=True)
    vectors = embedding_model.embed_documents(docs)
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors).astype('float32'))
    faiss.write_index(index, os.path.join(save_path, "index.faiss"))
    with open(os.path.join(save_path, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)

def load_vector_store(path="vectorstore/jd"):
    index = faiss.read_index(os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "docs.pkl"), "rb") as f:
        docs = pickle.load(f)
    return index, docs
