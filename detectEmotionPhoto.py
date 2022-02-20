from deepface import DeepFace
import json
import cv2
import helper
import numpy as np
np.set_printoptions(suppress=True)

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread('ExampleImages/justin.jpg')
# Must convert to greyscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
# Detect Faces 
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

img_crop = []

# Draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
    img_crop.append(img[y:y+h, x:x+w])

avg_array = np.zeros(7)

for cropped in img_crop:
    #cv2.imshow('Cropped', cropped)
    obj = DeepFace.analyze(img_path = cropped, actions = ['emotion', 'race', 'age', 'gender'], enforce_detection=False)
    print(obj)
    #print(obj['dominant_emotion'])
    helper.addToAverages(avg_array, obj)
    #cv2.waitKey(0)

print(avg_array/len(img_crop))
#'angry': 20.55305689573288, 'disgust': 0.013557590136770159, 'fear': 35.51655411720276, 'happy': 3.986029699444771, 'sad': 9.427881985902786, 'surprise': 0.7301448844373226, 'neutral'