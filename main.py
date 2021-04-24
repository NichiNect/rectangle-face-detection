import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trainedFaceData =  cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

# Choose image to detect for example then resize
img = cv2.imread('data-test/pic2.jpg')
imgResize = cv2.resize(img, (820, 720))

# Change to GrayScale
filterGrayScaled = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)

# Detect faces
coordinateFaces = trainedFaceData.detectMultiScale(filterGrayScaled)

# Draw the rectangle
for (x, y, width, height) in coordinateFaces: 
    rectangledImg = cv2.rectangle(imgResize, (x, y), (x + width, y + height), (0, 255, 0), 3)

print(coordinateFaces)

# Show the Face
cv2.imshow('Face Detection - Yoni Widhi', rectangledImg)
cv2.waitKey(0)

print('Hello Code!')