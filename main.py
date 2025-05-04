from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
import os
import shutil
import numpy as np
import face_recognition
from PIL import Image, UnidentifiedImageError
import io
from typing import List, Dict, Set
import uvicorn

app = FastAPI()

BASE_DIR = "uploaded_images"

# Ensure base directory exists
os.makedirs(BASE_DIR, exist_ok=True)

faculty_assignments: Dict[tuple, Set[str]] = {}  # Dictionary to track faculties assigned to folders

def get_org_folder_path(org_name: str, folder_name: str):
    return os.path.join(BASE_DIR, org_name, folder_name)

def load_student_encodings(folder_path: str):
    """Load student images from a given folder and return encodings."""
    students = {}
    if not os.path.exists(folder_path):
        return {"error": f"Folder '{folder_path}' does not exist"}

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                students[filename.split(".")[0]] = encoding[0]

    return students

def process_image(image: np.ndarray, students: dict):
    """Helper function to compare uploaded image with stored student images."""
    group_face_encodings = face_recognition.face_encodings(image)
    if not group_face_encodings:
        return {"error": "No faces detected in the image"}

    present_students = set()
    for group_encoding in group_face_encodings:
        for name, student_encoding in students.items():
            match = face_recognition.compare_faces([student_encoding], group_encoding, tolerance=0.6)
            if match[0]:
                present_students.add(name)

    missing_students = set(students.keys()) - present_students
    return {"missing_students": list(missing_students)}

@app.post("/create_folder/")
async def create_folder(org_name: str = Form(...), folder_name: str = Form(...), files: List[UploadFile] = File(...)):
    """Creates a new folder under an organization and stores images."""
    org_path = os.path.join(BASE_DIR, org_name)
    folder_path = os.path.join(org_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    for file in files:
        file_path = os.path.join(folder_path, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    return {"message": f"Folder '{folder_name}' created under organization '{org_name}' with {len(files)} images."}

@app.post("/check_attendance/")
async def check_attendance(org_name: str = Form(...), folder_name: str = Form(...), file: UploadFile = File(...)):
    """Check attendance using an uploaded image, ensuring only assigned faculty can check."""
    folder_path = get_org_folder_path(org_name, folder_name)

    # Load student encodings from the specified folder
    students = load_student_encodings(folder_path)
    if "error" in students:
        return students

    try:
        # Read and process uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
        image_array = np.array(image)

        return process_image(image_array, students)

    except UnidentifiedImageError:
        return {"error": "Invalid image format. Please upload a valid image file (JPG or PNG)."}
    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}

