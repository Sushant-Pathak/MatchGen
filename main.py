import os
from dotenv import load_dotenv
from utils.drive_handler import authenticate_gdrive, download_files_from_folder
from utils.resume_parser import parse_resume
from utils.jd_parser import parse_jd
from vectorstore.vector_store import create_vector_store, load_vector_store
from vectorstore.vector_store_res import create_vector_store_res, load_vector_store_res
from chains.resume_matcher import compute_similarity

load_dotenv()

# Config
RESUME_FOLDER_ID = os.getenv("RESUME_FOLDER_ID")
JD_FOLDER_ID = os.getenv("JD_FOLDER_ID")
resume_dir = "data/resumes"
jd_dir = "data/job_descriptions"

# Authenticate Google Drive
service = authenticate_gdrive()

# Download files
download_files_from_folder(service, RESUME_FOLDER_ID, resume_dir)
download_files_from_folder(service, JD_FOLDER_ID, jd_dir)

# Parse text
resume_texts = [parse_resume(os.path.join(resume_dir, f)) for f in os.listdir(resume_dir)]
jd_texts = [parse_jd(os.path.join(jd_dir, f)) for f in os.listdir(jd_dir)]

#print("jd_text :", jd_texts)
#print("resume_text:",resume_texts)

# Create vector store from JDs
create_vector_store(jd_texts)
create_vector_store_res(resume_texts)

# Load vector store
indexjd, jd_docs = load_vector_store()
indexres,res_docs=load_vector_store_res()


# Match resumes
for i, resume_text in enumerate(resume_texts):
    print(f"\n===== Resume {i + 1} Matching Results =====\n")
    matches = compute_similarity(resume_text, indexjd, jd_docs)
    for j, (jd_text, score) in enumerate(matches):
        print(f"Job Description {j + 1} - Match Score: {score:.4f}")
         # Print first 300 characters of the JD
        print("-" * 40)

