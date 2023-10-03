import numpy
import cv2
import os

import numpy as np
import pylab as pl
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import utils
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam

import pickle

###########################################

path = "E:\AI - ML Projects\TextDetection_using_NN_CV\myData"
pathLabels = 'labels.csv'
test_ratio = 0.2
validation_ratio = 0.2
imageDimensions = (32, 32, 3)

batchSizeVal = 50
epochsVal = 10
# stepsPerEpochVal = 2000

###########################################

#1 first, we'll get the list of directories in our path (names of the each of the folder)
images = []
classNo = [] # Contains IDs of all images
myList = os.listdir(path)
# print(myList)
print("Total no of classes detected: ", len(myList))
noOfClasses = len(myList)
print("Importing classes....")


#2 To read and store images
for x in range(noOfClasses):
    picList = os.listdir(path + "/" + str(x))
    for y in picList:
        curImg = curImg = cv2.imread(os.path.join(path, str(x), y))
        ## Our image dimension is 180x180 which is so computationally expensive. So, we have to resize the image
        curImg = cv2.resize(curImg, (imageDimensions[0], imageDimensions[1]))
        # curImg = cv2.resize(curImg, (32, 32))
        images.append(curImg)

        # Next, we have to save the corresponding ID which is the class ID of each of these images
        classNo.append(x)
    print(x, end=" ")
print("\nNo. of images: ", len(images))


#3 Now, next we will convert it into numpy array
images = np.array(images)
classNo = np.array(classNo) # IDs of all images

print(images.shape)
print(classNo.shape)


#4 Splitting the data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=test_ratio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_ratio)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(X_validation.shape)


#5 finding IDs who have class = 0 (e.g.)
# print(np.where(y_train==0))
noOfSamples = []
for x in range(0, noOfClasses):
    noOfSamples.append(len(np.where(y_train==x)[0]))
    # It gives, how many images we have for each number
# print(noOfSamples)


#6 Making barplot
plt.figure(figsize=(10,5))
plt.bar(range(0, noOfClasses), noOfSamples)
plt.title("No of Images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of images")
plt.show()


#7 Preprocessing
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)  # it makes the lighting of the image distribute evenly
    img = img/255  # normalize image

    return img

# To show the image
# img = preProcessing(X_train[50])
# img = cv2.resize(img, (200,200))
# cv2.imshow("Preprocessing ", img)
# cv2.waitKey(0)

# img = X_train[50]
# img = cv2.resize(img, (200,200))
# cv2.imshow("Preprocessed ", img)
# cv2.waitKey(0)

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))


#8 Add depth --> It's imp for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation= X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)


#9 Now we augment our images --> Means we'll add zoom, some rotation, will shift, add some translation
#  This will make dataset more generic
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

#10 One hot encoding
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

# This model basically based on LeNet model
def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNode = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0], imageDimensions[1],1),
                                                  activation = 'relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size = sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size = sizeOfPool))
    model.add(Dropout(0.5)) # means 50% --> Dropout layer is helping you to reduce the overfitting

    model.add(Flatten())
    model.add(Dense(noOfNode, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation = 'softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = myModel()
print(model.summary())

stepsPerEpochVal = len(X_train) // batchSizeVal

#11 to run the training
history = model.fit(dataGen.flow(X_train, y_train,
                                     batch_size=batchSizeVal),
                                     steps_per_epoch = stepsPerEpochVal,
                                     epochs = epochsVal,
                                     validation_data = (X_validation, y_validation),
                                     shuffle = 1)

#12 Now, after the training process we want to know how was the loss of variation in the loss and accuracy
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('no. of epochs')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('no. of epochs')
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])


#13 Save the model
pickle_out = open("model_trained_10.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()

