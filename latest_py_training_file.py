import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face_LBPHFaceRecognizer.create()  # Create the face recognizer
path = "dataset"

def get_images_with_id(path):
    images_path = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for single_image_path in images_path:
        faceImg = Image.open(single_image_path).convert('L')
        faceNp = np.array(faceImg, np.uint8)
        id = int(os.path.split(single_image_path)[-1].split(".")[1])
        print(id)
        faces.append(faceNp)
        ids.append(id)
        cv2.imshow("Training", faceNp)
        cv2.waitKey(10)

    return np.array(ids), faces

ids, faces = get_images_with_id(path)
recognizer.train(faces, ids)
recognizer.save("recognizer/trainingset.yml")
cv2.destroyAllWindows()
