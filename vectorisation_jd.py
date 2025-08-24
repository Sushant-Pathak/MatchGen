import os
from typing import List, Tuple
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Convert JD → Document (direct, no corpus splitting)
def jd_to_document(jd_info, jd_id, title):
    """
    Convert parsed JD (ResumeInfo object) to a LangChain Document
    """
    # Convert JD structured object to string for vectorization
    jd_text = jd_info.model_dump_json()  # Or use model_dump() for dict
    return Document(
        page_content=jd_text,  # Now it's a string
        metadata={"resume_id": jd_id, "title": title}
    )

# Store JDs in FAISS
def store_jds_in_faiss(jds: List[Tuple[str, str, str]], faiss_path="jd_index"):
    """
    Store multiple JDs into FAISS vector DB.
    jds: list of (jd_id, title, jd_text)
    """
    all_docs = []

    for jd_id, title, jd_text in jds:
        doc = jd_to_document(jd_text, jd_id, title)
        all_docs.append(doc)

    embeddings = OpenAIEmbeddings()

    # If FAISS DB already exists, load and add new docs
    if os.path.exists(faiss_path):
        db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(all_docs)
    else:
        db = FAISS.from_documents(all_docs, embeddings)

    # Save database
    db.save_local(faiss_path)
    print(f"Stored {len(all_docs)} JDs into FAISS at {faiss_path}")
