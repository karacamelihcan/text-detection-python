import numpy as np
import cv2
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

##############Project Settings###################
width = 640
height = 4480
threshold = 0.7
#################################################

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)


def preProcessing(img):  # resim alan bir fonksiyon tanımladık
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # resmi gray-scale hale çevirdik.
    img = cv2.equalizeHist(img)  # görüntülerin aydınlatmasının eşit dağılmasını sağlar
    # Normalization aşaması. Grayscale değerleri 0-255 arasında değişir. Biz bunları 0-1 arasında olmasını sağlıyoruz. Training
    # için böyle olması daha iyi
    img = img / 255
    return img

while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)

    #Predict
    classIndex = int(model.predict_classes(img))
    print(classIndex)
    predictions = model.predict(img)
    #print(predictions)
    probValue = np.amax(predictions)
    print(classIndex, probValue)
    if probValue > threshold:
        cv2.putText(imgOriginal, str(classIndex)+"  " + str(probValue), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow("Original Image", imgOriginal)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
