import cv2
import sys

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
if len(sys.argv) > 1:
    if sys.argv[1]=='eye': 
        trainedFaceData =  cv2.CascadeClassifier('haarcascade_eye.xml')
else:
    trainedFaceData =  cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

# Choose image to detect for example then resize
cam = cv2.VideoCapture(0)
# imgResize = cv2.resize(cam, (820, 720))

# Iterate the camera's frame
while True:

    successful_frame_read, frame = cam.read()

    # Change to GrayScale
    filterGrayScaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    coordinateFaces = trainedFaceData.detectMultiScale(filterGrayScaled)

    # Draw the rectangle
    for (x, y, width, height) in coordinateFaces: 
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)

    # print(coordinateFaces)

    # Show the Face
    cv2.imshow('Face Detection - Yoni Widhi', frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

cam.release()
print('Hello Code!')