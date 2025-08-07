import os
import faiss
import pickle
import numpy as np
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables for OpenAI
load_dotenv()

# Initialize embedding model
embedding_model = OpenAIEmbeddings()

# Create vector store specifically for resumes
def create_vector_store_res(docs, save_path="vectorstore/res"):
    os.makedirs(save_path, exist_ok=True)
    
    # Embed all resume documents
    vectors = embedding_model.embed_documents(docs)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors).astype('float32'))
    
    # Save index and docs
    faiss.write_index(index, os.path.join(save_path, "index.faiss"))
    with open(os.path.join(save_path, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)

# Load resume vector store
def load_vector_store_res(path="vectorstore/res"):
    index = faiss.read_index(os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "docs.pkl"), "rb") as f:
        docs = pickle.load(f)
    return index, docs
