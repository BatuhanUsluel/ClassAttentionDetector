import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
import json
import helper
import numpy as np
np.set_printoptions(suppress=True)

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

vidcap = cv2.VideoCapture('ExampleVideo/Zoom Video.mp4')
success,img = vidcap.read()
count = 0
while success:
  #cv2.imwrite("frame%d.jpg" % count, img)     # save frame as JPEG file      
  success,img = vidcap.read()
  print(helper.processFrame(img))
  print('Read a new frame: ', success)
  count += 1