import cv2
import numpy as np
import sqlite3

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the camera
cam = cv2.VideoCapture(0)  # 0 is for the default webcam

def insert_or_update(Id, Name, age):
    # Connect to the SQLite database
    conn = sqlite3.connect("SQLite.db")
    cmd = f"SELECT * FROM PEOPLE WHERE ID={Id}"
    cursor = conn.execute(cmd)
    is_record_exist = 0  # Assume there is no record in our table

    for row in cursor:
        is_record_exist = 1

    if is_record_exist == 1:
        # If a record exists, update the name and age
        conn.execute("UPDATE PEOPLE SET Name=? WHERE ID=?", (Name, Id))
        conn.execute("UPDATE PEOPLE SET age=? WHERE ID=?", (age, Id))
    else:
        # If no record exists, insert the values
        conn.execute("INSERT INTO PEOPLE (Id,Name,age) VALUES (?,?,?)", (Id, Name, age))

    conn.commit()
    conn.close()

# Insert user-defined values into the table
Id = input('Enter User Id: ')
Name = input('Enter User Name: ')
age = input('Enter User Age: ')

insert_or_update(Id, Name, age)

# Detect faces in the webcam feed
sampleNum = 0  # Assume there are no samples in the dataset
while True:
    ret, img = cam.read()  # Open the camera
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)  # Detect faces

    for (x, y, w, h) in faces:
        sampleNum += 1  # Increment if a face is detected
        cv2.imwrite(f"dataset/user.{Id}.{sampleNum}.jpg", gray[y:y + h, x:x + w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.waitKey(100)  # Delay time

    cv2.imshow("Face", img)  # Show faces detected in the webcam
    cv2.waitKey(1)

    if sampleNum > 20:  # If the dataset has more than 20 samples, break
        break

cam.release()
cv2.destroyAllWindows()  # Close all windows