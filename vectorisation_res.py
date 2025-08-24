import os
from typing import List, Tuple
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from seperator import ResumeInfo

def resume_to_corpus(resume: ResumeInfo, resume_id: str, title: str) -> List[Document]:     #  Convert Resume to Corpus
    # Convert ResumeInfo into a single corpus Document with metadata including resume_id and title.
    corpus_parts = []

    # Skills
    if resume.skills:
        corpus_parts.append("Skills: " + ", ".join(resume.skills))

    # Education
    if resume.education:
        corpus_parts.append("Education: " + ", ".join(resume.education))

    #  Projects
    if resume.projects:
        corpus_parts.append("Projects: " + ", ".join(resume.projects))

    #  Project Skills
    if resume.project_skills:
        corpus_parts.append("Project Skills: " + ", ".join(resume.project_skills))

    #  Achievements
    if resume.achievements:
        corpus_parts.append("Achievements: " + ", ".join(resume.achievements))

    #  Experience
    if resume.experience_no_year:
        for exp in resume.experience_no_year:
            exp_text = f"Experience at {exp.company} for {exp.duration}"
            if exp.skilss:
                exp_text += f" | Skills: {', '.join(exp.skilss)}"
            if exp.impact_works:
                exp_text += f" | Impact: {', '.join(exp.impact_works)}"
            corpus_parts.append(exp_text)

    # Join into one big block of text
    corpus_text = "\n".join(corpus_parts)

    return [
        Document(
            page_content=corpus_text,
            metadata={"resume_id": resume_id, "title": title}
        )
    ]


#  Store Resumes in FAISS
def store_resumes_in_faiss(resumes: List[Tuple[str, str, ResumeInfo]], faiss_path="resume_index"):

    # Store multiple resumes into FAISS vector DB. resumes: list of (resume_id, title, ResumeInfo)

    all_docs = []

    for resume_id, title, resume_data in resumes:
        docs = resume_to_corpus(resume_data, resume_id, title)
        all_docs.extend(docs)

    embeddings = OpenAIEmbeddings()

    # If FAISS DB already exists, load and add new docs
    if os.path.exists(faiss_path):
        db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(all_docs)
    else:
        db = FAISS.from_documents(all_docs, embeddings)

    # Save database
    db.save_local(faiss_path)
    print(f"Stored {len(all_docs)} resumes into FAISS at {faiss_path}")