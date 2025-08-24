import os
from dotenv import load_dotenv
from google.oauth2 import service_account
from langchain_google_community import GoogleDriveLoader
from seperator import parse_resume_text
from vectorisation_res import store_resumes_in_faiss
from vectorisation_jd import store_jds_in_faiss
from semantic import generate_similarity_matrix

def main():

    #loading the apis and paths from .env
    load_dotenv()

    CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH")
    TOKEN_PATH = os.getenv("GOOGLE_TOKEN_PATH") or "token.json"
    RES_FOLDER_ID = os.getenv("GOOGLE_DRIVE_RES_FOLDER_ID")
    JD_FOLDER_ID = os.getenv("GOOGLE_DRIVE_JD_FOLDER_ID")

    # check for values if path exist or not 
    if not CREDENTIALS_PATH or not RES_FOLDER_ID:
        raise ValueError(" Missing GOOGLE_CREDENTIALS_PATH or GOOGLE_DRIVE_RES_FOLDER_ID in .env")

    print("Environment variables loaded successfully")
    print("   Using credentials:", CREDENTIALS_PATH)
    print("   Using token file: ", TOKEN_PATH)
    print("   Resume Drive folder ID:  ", RES_FOLDER_ID)
    print("   JD Drive folder ID:  ", JD_FOLDER_ID)

    # Load service account credentials
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )

    # Initialize loaders
    res_loader = GoogleDriveLoader(
        folder_id=RES_FOLDER_ID,
        credentials=creds,
        token_path=TOKEN_PATH,
        recursive=True,
        file_types=["application/pdf"],
        load_auth=False
    )

    jd_loader = GoogleDriveLoader(
        folder_id=JD_FOLDER_ID,
        credentials=creds,
        token_path=TOKEN_PATH,
        recursive=True,
        file_types=["application/pdf"],
        load_auth=False
    )

    # Load resumes
    res_files = res_loader.load()
    if not res_files:
        print("No resumes found. Check folder ID, permissions, or file_types.")
        return
    print(f"Found {len(res_files)} resume(s) in Drive folder.")

    # Load JDs
    jd_files = jd_loader.load()
    if not jd_files:
        print("No JDs found. Check folder ID, permissions, or file-types.")
        return
    print(f"Found {len(jd_files)} JD(s) in Drive folder.")

    # Process resumes
    resumes_for_db = []
    for i, doc in enumerate(res_files, start=1):
        title = doc.metadata.get("title", f"Resume {i}")
        resume_id = doc.metadata.get("source", f"resume_{i}")
        text = doc.page_content

        print(f"\n--- Processing Resume {i}: {title} ---")
        try:
            resume_info = parse_resume_text(text)
            resumes_for_db.append((resume_id, title, resume_info))
            print("Parsed successfully and added for FAISS storage.")
        except Exception as e:
            print(f"Error parsing {title}: {e}")

    if resumes_for_db:
        store_resumes_in_faiss(resumes_for_db)
    else:
        print("No resumes parsed successfully. Skipping FAISS storage.")

    # Process JDs
    jds_for_db = []
    for i, doc in enumerate(jd_files, start=1):
        title = doc.metadata.get("title", f"JD {i}")
        jd_id = doc.metadata.get("source", f"jd_{i}")
        text = doc.page_content

        print(f"\n--- Processing JD {i}: {title} ---")
        try:
            jd_info = parse_resume_text(text)
            jds_for_db.append((jd_id, title, jd_info))
            print("Parsed successfully and added for FAISS storage.")
        except Exception as e:
            print(f"Error parsing {title}: {e}")

    if jds_for_db:
        store_jds_in_faiss(jds_for_db)
    else:
        print("No JDs parsed successfully. Skipping FAISS storage.")

    
    # similarity score csv generator
    generate_similarity_matrix(
        faiss_resume_path="resume_index",
        faiss_jd_path="jd_index",
        output_csv="final_similarity.csv"
    )




if __name__ == "__main__":
    main()
