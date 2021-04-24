import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trainedFaceData =  cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print('Hello Code!')