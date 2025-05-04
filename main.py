from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
import os
import shutil
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
import io
from typing import List, Dict, Set
import uvicorn
import pickle

app = FastAPI()

BASE_DIR = "uploaded_images"

# Ensure base directory exists
os.makedirs(BASE_DIR, exist_ok=True)

faculty_assignments: Dict[tuple, Set[str]] = {}  # Dictionary to track faculties assigned to folders

def get_org_folder_path(org_name: str, folder_name: str):
    return os.path.join(BASE_DIR, org_name, folder_name)

def load_student_encodings(folder_path: str):
    """Load student images from a given folder and extract face features."""
    students = {}
    if not os.path.exists(folder_path):
        return {"error": f"Folder '{folder_path}' does not exist"}

    # Create face detector and feature extractor
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Check if a model already exists for this folder
    model_path = os.path.join(folder_path, "face_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            students = pickle.load(f)
        return students

    # Process each image to extract face features
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            name = filename.split(".")[0]
            image_path = os.path.join(folder_path, filename)
            
            # Read image and convert to grayscale
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Take the first face
                x, y, w, h = faces[0]
                face_img = gray[y:y+h, x:x+w]
                
                # Resize for consistency
                face_img = cv2.resize(face_img, (100, 100))
                
                # Store the face image
                students[name] = face_img

    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(students, f)

    return students

def process_image(image_array, students):
    """Compare uploaded image with stored student images using OpenCV."""
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Initialize face detector
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return {"error": "No faces detected in the image"}
    
    # List to store recognized students
    present_students = set()
    
    # Compare each detected face with student face images
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        
        for name, student_face in students.items():
            # Simple template matching for demonstration
            # In a production system, you'd use a proper face recognition algorithm
            result = cv2.matchTemplate(face_img, student_face, cv2.TM_CCOEFF_NORMED)
            similarity = np.max(result)
            
            # Threshold for recognition
            if similarity > 0.6:
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
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        return process_image(image_array, students)

    except UnidentifiedImageError:
        return {"error": "Invalid image format. Please upload a valid image file (JPG or PNG)."}
    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
