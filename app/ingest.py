# =============================================================================
# ingest.py
# PURPOSE: Read courses.json, convert each course into a LangChain Document,
#          embed them using HuggingFace, and save the FAISS vector index locally.
# RUN THIS ONCE (or whenever courses.json changes) before starting advisor.py
# =============================================================================

import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def create_vector_db():
    json_path = '../Data/courses.json'

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Please check the Data folder.")
        return

    with open(json_path, 'r') as f:
        courses = json.load(f)

    documents = []

    for course in courses:
        outcomes_list = course.get('course_outcomes', [])
        topics_list   = course.get('topics', [])

        outcomes_text = " ".join(outcomes_list)
        topics_text   = " ".join(topics_list)

        if not outcomes_list:
            print(f"Warning: No outcomes found for {course.get('course_code')}")

        # -----------------------------------------------------------------
        # IMPORTANT: course_code MUST be in page_content so FAISS can find
        # it when a user queries a specific code like "BCS613B".
        # Previously the code was missing a space/period after course_code
        # which caused it to merge with the course name during embedding.
        # -----------------------------------------------------------------
        page_content = (
            f"Course Code: {course['course_code']}. "   # <-- fixed: added period+space
            f"Course: {course['name']}. "
            f"Description: {course['description']}. "
            f"Topics: {topics_text}. "
            f"Outcomes: {outcomes_text}"
        )

        metadata = {
            "course_id"    : course['course_code'],
            "name"         : course['name'],
            "prerequisites": course.get('prerequisites', []),
            "credits"      : course.get('credits', 0),
            "outcomes"     : outcomes_text
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    print(f"Total courses loaded: {len(documents)}")
    print("Initializing embedding model...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Building the FAISS vector database...")
    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local("faiss_index")

    print("Done. faiss_index folder has been created successfully.")


if __name__ == "__main__":
    create_vector_db()