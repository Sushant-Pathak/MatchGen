import numpy as np
from langchain_openai import OpenAIEmbeddings
from vectorstore.vector_store import load_vector_store

embedding_model = OpenAIEmbeddings()

def compute_similarity(resume_text, index, jd_texts):
    query_vec = embedding_model.embed_query(resume_text)
    D, I = index.search(np.array([query_vec]).astype('float32'), k=len(jd_texts))
    return [(jd_texts[i], float(1 - D[0][idx])) for idx, i in enumerate(I[0])]
