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

# Resimleri tutmka için bir liste oluşturduk
images = []
classNo = []

# 1) Öncelikle listemizin directory bilgisini alalım
myList = os.listdir(path)
# print(myList) listemizin içi
print("Total Number of Classes Detected: ", len(myList))  # listemizin boyutu 2
numberOfClasses = len(myList)

# bütün resimleri import edip bunları bir listeye koyalım
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
# print(len(images)) # 4.foto listemize eklediğimiz resimlerin sayısı
# print(len(classNo)) # 5. Foto bütün resimleri class numaraları ile birlikte tuttuğumuz için class no listesinin sayısı


# Elde ettiğimiz listeleri numpy arraye çevirelim
images = np.array(images)
classNo = np.array(classNo)

# Shapeleri kontrol etmek daha kolay 6 .foto
print(images.shape)
# print(classNo.shape)

# Split Data işlemini yapabilmek için sklearn kütüphanesini kullanacağız.
X_train, X_test, Y_train, Y_Test = train_test_split(images, classNo, test_size=testRatio)  # %20 test %80 training
# Validation yapabilmek için tekrar split işlemi yapalım
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train,
                                                                test_size=validationRatio)  # %64 Train %16 Validation

print(X_train.shape)  # Split edilip edilmediğini kontrol edelim foto 7
print(X_test.shape)
print(X_validation.shape)  # Validataion ı split ettikten sonrası foto 8

# Verilerin her sınıfa eşit dağıldığından emin olmalyız. X_train asıl resimleri tutarken Y_train her resmin sınıf bilgisini tutar
# print(np.where(Y_train == 0)) # 0 sınıfını temsil eden indexler foto 9-1 ve 9-2

# print(len(np.where(Y_train == 0)[0])) # ID si 0 olanların sayısı foto 10
# bu işlemi tamamı için yapalım
# HEr sınıfın kaç tane resmi olduğununun sayısı foto 11
# for x in range(0, numberOfClasses):
#    print(len(np.where(Y_train == x)[0]))

# her sınıftan kaç tane olduğunu bir değişkene atayalım

numberOfSamples = []
for x in range(0, numberOfClasses):
    # print(len(np.where(Y_train == x)[0]))
    numberOfSamples.append(len(np.where(Y_train == x)[0]))
print(numberOfSamples)  # her sınıftan kaç tane örnek var onu bastırıyoruz foto 12

# Resimlerin sınıflara nasıl dağıldığına dair bir grafik oluşturalım. Bunun için matplotlib i import ettik
plt.figure(figsize=(10, 5))
plt.bar(range(0, numberOfClasses), numberOfSamples)
plt.title("Number of Images for Each Class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()  # foto 13

print("Before Preprocess:")
print(X_train[44].shape)


# Sıradaki işlem resimleri pre process etmek
def preProcessing(img):  # resim alan bir fonksiyon tanımladık
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # resmi gray-scale hale çevirdik.
    img = cv2.equalizeHist(img)  # görüntülerin aydınlatmasının eşit dağılmasını sağlar
    # Normalization aşaması. Grayscale değerleri 0-255 arasında değişir. Biz bunları 0-1 arasında olmasını sağlıyoruz. Training
    # için böyle olması daha iyi
    img = img / 255
    return img


# # Herhangi bir resmimizin pre-process işleminden sonra nasıl göründüğü test edelim. Seçtiğim resmim Traing setinin 44 nolu resmi
# img = preProcessing(X_train[44])
# imgOriginal = X_train[44]
# # resimler daha önce 32*32 boyutunda olduğu için görmek için resize edelim
# img = cv2.resize(img, (300, 300))
# imgOriginal = cv2.resize(imgOriginal, (300, 300))
# cv2.imshow("Non-PreProcessed", imgOriginal)
# cv2.imshow("PreProcessed", img)
# cv2.waitKey(0) # Hemen kapanmamasını sağladık


# Şimdi yapmamız gereken X_Train arrayi içerisinde bulunan bütün resimleri Pre-process işlemi uygulamalıyız

X_train = np.array(list(map(preProcessing, X_train)))
# img = X_train[44]
# img = cv2.resize(img, (300, 300))  foto 15
# cv2.imshow("PreProcessed", img)
# cv2.waitKey(0)

# Şimdi Pre-process işleminden sonra herhangi bir resmin shape'ine bakalım
# print("After preprocess:")  # foto 16
# print(X_train[44].shape)

# Preprocess işlemini şimdide Test ve Validation setleri için uygulayalım
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

# # sonraki adım, resimlerimize bir depth eklemek. Sinir ağlarının düzgün çalışabilmesi için bu gerekli
# print("Before the reshape")
# print(X_train.shape)
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
# print("After the reshape")
# print(X_train.shape) # foto 17

# Bu işlemi hepsi için yapmalıyız
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

# Datasetimizi daha generic bir hale getirmeye çalışacağız. Kerası kullanacağız. Augment Images
dataGenerator = ImageDataGenerator(width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2,
                                   shear_range=0.1,
                                   rotation_range=10)

dataGenerator.fit(X_train)

# from keras.utils.np_utils import to_categorical ekldik
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


# history = model.fit_generator(dataGenerator.flow(X_train, Y_train, batch_size=batchSizeValue),
#                                     steps_per_epoch=stepsPerEpoch,
#                                     epochs=epochsValue,
#                                     validation_data=(X_validation,Y_validation),
#                                     shuffle=1)

history = model.fit(dataGenerator.flow(X_train, Y_train, batch_size=batchSizeValue),
                                    steps_per_epoch=stepsPerEpoch,
                                    epochs=epochsValue,
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



