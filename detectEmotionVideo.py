import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
import json
import helper
import numpy as np
import collections

np.set_printoptions(suppress=True)

vidcap = cv2.VideoCapture('ExampleVideo/Zoom Video.mp4')
success,img = vidcap.read()
count = 0

emotions = collections.deque(np.zeros(shape=(20,7)))

while success:
  #cv2.imwrite("frame%d.jpg" % count, img)     # save frame as JPEG file      
  success,img = vidcap.read()
  print(helper.processFrame(img))
  print(f'Read frame count {count} with success: {success}')
  count += 1