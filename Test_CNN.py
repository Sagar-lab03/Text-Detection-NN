# # # ---- Testing Script ----- # # #

import numpy as np
import cv2
import pickle

##########################################
width = 640
height = 480
threshold = 0.65
########################################################

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Next, we unpickle all our objects and we're going to put it in the variable model
pickle_in = open("model_trained_10.p", "rb")
model = pickle.load(pickle_in)

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)  # it makes the lighting of the image distribute evenly
    img = img/255  # normalize image

    return img

while True:
    success, imgOriginal = cap.read()

    # Check if image is captured successfully
    if not success:
        print("Failed to grab frame")
        continue

    # Assert that imgOriginal is not None
    # assert imgOriginal is not None, "imgOriginal is None"

    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    # cv2.imshow("Preprocessed Image", img)
    img = img.reshape(1,32,32,1)

    # Predict
    # classIndex = int(model.predict_classes(img))
    prediction = model.predict(img)
    classIndex = np.argmax(prediction, axis=-1)
    probVal = np.amax(prediction)
    print(classIndex, probVal)

    if probVal > threshold:
        cv2.putText(imgOriginal, str(classIndex) + "  " + str(probVal), (50,50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0,0,255), 1)

    cv2.imshow("Original Image", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




