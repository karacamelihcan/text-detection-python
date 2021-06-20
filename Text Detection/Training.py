import numpy as np
import cv2
import os
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Flatten, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
import pickle

##############Project Settings###################
path = "Dataset"
testRatio = 0.2
validationRatio = 0.2
imageDimensions = (32, 32, 3)
batchSizeValue = 50
epochsValue = 10
stepsPerEpoch = 2000
#################################################

images = []
classNo = []

myList = os.listdir(path)
print("Total Number of Classes Detected: ", len(myList))
numberOfClasses = len(myList)

print("Importing Classes.....")
for x in range(0, numberOfClasses):
    myPictureList = os.listdir(
        path + "/" + str(x))  # Sırasıyla bütün klasörlerin içini okuyacağımız için dizini belirttik
    for y in myPictureList:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)
        # Veri setimizdeki resimler büyük olduğundan dolayı yeniden boyutlandıralım
        curImg = cv2.resize(curImg, (imageDimensions[0], imageDimensions[1]))
        # Yeniden boyutlandırdığımz resmi oluşturduğumuz listeye ekleyelim.
        images.append(curImg)
        # Bu resimleri ilgili sınıf id leri ile beraber saklamalıyız.
        classNo.append(x)
    print(x, end=" ")  # Hangi sınıfları eklediğimizi kontrol edelim 3. Foto
print(" ")
print(len(images))
print(len(classNo))


# Elde ettiğimiz listeleri numpy arraye çevirelim
images = np.array(images)
classNo = np.array(classNo)
print(images.shape)


# Split Data işlemini yapabilmek için sklearn kütüphanesini kullanacağız.
X_train, X_test, Y_train, Y_Test = train_test_split(images, classNo, test_size=testRatio)  # %20 test %80 training
# Validation yapabilmek için tekrar split işlemi yapalım
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train,
                                                                test_size=validationRatio)  # %64 Train %16 Validation

print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

# her sınıftan kaç tane örnek var
numberOfSamples = []
for x in range(0, numberOfClasses):
    numberOfSamples.append(len(np.where(Y_train == x)[0]))
print(numberOfSamples)

# Resimlerin sınıflara nasıl dağıldığına dair bir grafik oluşturalım.
plt.figure(figsize=(10, 5))
plt.bar(range(0, numberOfClasses), numberOfSamples)
plt.title("Number of Images for Each Class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()  # foto 13

print("Before Preprocess:")
print(X_train[44].shape)


# Sıradaki işlem resimleri pre process etmek
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    # Normalization
    img = img / 255
    return img

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))



X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

# Datasetimizi daha generic bir hale getirmeye çalışacağız.
dataGenerator = ImageDataGenerator(width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2,
                                   shear_range=0.1,
                                   rotation_range=10)

dataGenerator.fit(X_train)

# One Hot Encoding
Y_train = to_categorical(Y_train, numberOfClasses)
Y_Test = to_categorical(Y_Test, numberOfClasses)
Y_validation = to_categorical(Y_validation, numberOfClasses)


def myModel():
    numberOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    numberOfNode = 500

    model = Sequential()
    # İlk katmanımızı oluşturduk
    model.add((Conv2D(numberOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0],
                                                                   imageDimensions[1],
                                                                   1), activation="relu")))

    # Başka bir katman daha oluşturduk. Dimension'ı tekrar eklememize gerek olmadığı için çıkardık
    model.add((Conv2D(numberOfFilters, sizeOfFilter1, activation="relu")))

    # Pooling Layer'ı ekliyoruz.
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    #İki tane daha convolutional layer ekledik. İkisinde de number of filters sayısını eksiltiyoruz. Size of Filter' ıda küçülttük
    model.add((Conv2D(numberOfFilters//2, sizeOfFilter2, activation="relu")))
    model.add((Conv2D(numberOfFilters//2, sizeOfFilter2, activation="relu")))
    #Pooling Layer ekledik,
    model.add(MaxPooling2D(pool_size=sizeOfPool))

    #İlk dropoput Layer'ı ekledik
    model.add(Dropout(0.5))

    #flattening Layer
    model.add(Flatten())

    #Dense Layer
    model.add(Dense(numberOfNode, activation="relu"))

    model.add(Dropout(0.5))

    model.add(Dense(numberOfClasses, activation="softmax"))

    model.compile(Adam(lr=0.001), loss="categorical_crossentropy", metrics=['accuracy'])

    return model

model = myModel()
print(model.summary()) #foto 18

# Son aşama run the training

history = model.fit(dataGenerator.flow(X_train, Y_train, batch_size=batchSizeValue),
                                    steps_per_epoch=stepsPerEpoch, #2000
                                    epochs=epochsValue, #10
                                    validation_data=(X_validation,Y_validation),
                                    shuffle=1)

# Loss ve Accuracy değerlerimizi plot şeklinde görsel hale getirdik
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title("Loss")
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title("Accuracy")
plt.xlabel('epoch')

plt.show()

score = model.evaluate(X_test, Y_Test, verbose=0)

print('Test Score =', score[0])
print('Test Accuracy =', score[1])

pickle_out = open("model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()



