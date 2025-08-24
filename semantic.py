import os
import csv
import numpy as np
from collections import OrderedDict
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def _group_and_aggregate_docs(docs, key_priority):
    groups = OrderedDict()
    display_names = {}
    for d in docs:
        md = d.metadata or {}
        grp_key = None
        for k in key_priority:
            v = md.get(k)
            if isinstance(v, str) and v.strip():
                grp_key = v.strip()
                break
        if grp_key is None:
            src = md.get("source") or md.get("path") or md.get("file_path") or md.get("filename") or ""
            grp_key = os.path.basename(src) if src else f"doc_{len(groups)}"
        if grp_key not in groups:
            groups[grp_key] = []
            display_names[grp_key] = (
                md.get("title")
                or os.path.basename(md.get("source") or md.get("filename") or grp_key)
                or grp_key
            )
        groups[grp_key].append(d.page_content or "")
    titles = [display_names[k] for k in groups.keys()]
    texts  = ["\n".join(groups[k]) for k in groups.keys()]
    keys   = list(groups.keys())
    return titles, texts, keys

def _norm(name: str) -> str:
    base = os.path.splitext(name)[0]
    return "".join(base.lower().split()).replace("_", "").replace("-", "")

def generate_similarity_matrix(
    faiss_resume_path: str = "resume_index",
    faiss_jd_path: str = "jd_index",
    output_csv: str = "jd_resume_similarity_matrix.csv"
):
    embeddings = OpenAIEmbeddings()

    resume_db = FAISS.load_local(faiss_resume_path, embeddings, allow_dangerous_deserialization=True)
    jd_db     = FAISS.load_local(faiss_jd_path, embeddings, allow_dangerous_deserialization=True)
    

    resume_docs = list(resume_db.docstore._dict.values())
    jd_docs     = list(jd_db.docstore._dict.values())

    # Group
    resume_titles, resume_texts, _ = _group_and_aggregate_docs(
        resume_docs, key_priority=["resume_id","filename","file_name","source","title"]
    )
    jd_titles, jd_texts, _ = _group_and_aggregate_docs(
        jd_docs, key_priority=["jd_id","filename","file_name","source","title"]
    )

    # Filter out accidental JDs inside resume index
    jd_norm_set = {_norm(t) for t in jd_titles}
    keep_idx = [i for i,t in enumerate(resume_titles) if _norm(t) not in jd_norm_set]
    resume_titles = [resume_titles[i] for i in keep_idx]
    resume_texts  = [resume_texts[i]  for i in keep_idx]

    if not resume_titles or not jd_titles:
        raise ValueError("No resumes or JDs left after filtering. Check your indexes/metadata.")

    # Embed
    resume_vecs = np.array(embeddings.embed_documents(resume_texts), dtype=np.float32)
    jd_vecs     = np.array(embeddings.embed_documents(jd_texts),     dtype=np.float32)

    def _safe_norm(x, axis=1, keepdims=True):
        n = np.linalg.norm(x, axis=axis, keepdims=keepdims)
        n[n == 0.0] = 1.0
        return n
    R = resume_vecs / _safe_norm(resume_vecs)
    J = jd_vecs     / _safe_norm(jd_vecs)
    matrix = R @ J.T   # (#resumes × #JDs)

    # Write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Resume/JD"] + jd_titles)
        for i, row in enumerate(matrix):
            writer.writerow([resume_titles[i]] + [f"{v:.4f}" for v in row])

    print(f"\nSimilarity matrix saved to {output_csv}")
    return matrix, resume_titles, jd_titles

if __name__ == "__main__":
    generate_similarity_matrix()
