import cv2
import numpy as np
import time
import keras
import requests
import imutils

model = keras.models.load_model('keras_modeltest.h5', compile=False)  
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32) # 數組對象表示一個多維、同質的固定大小項目數組
cap = cv2.VideoCapture(1) # 0:後鏡 1:前鏡

def lineNotify(fileNum):
    url = 'https://notify-api.line.me/api/notify'
    token = '3U0X4m8n0dJdCKmcF8OfGKe8it9347RY4J9APMju3CG' # 連續發送太多次會掛掉
    headers = {
        'Authorization': 'Bearer ' + token 
    }
    data = {
        'message':'Sunglasses detected!'
    }
    pic = open(r'C:\Users\littl\Desktop\sticker\photo_'+str(fileNum)+'.jpg','rb')

    files = {
        'imageFile':pic
    }
    r = requests.post(url, headers=headers, data=data, files=files)
    return r.status_code

if not cap.isOpened():
    print("Cannot open camera")
    exit()

fileNum = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    img = cv2.resize(frame , (398,224))
    img = img[0:224, 80:304]
    image_array = np.asarray(img) # 將列表轉換為數組 asarray不會佔用內存
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1 # 強製轉換為指定類型
    data[0] = normalized_image_array
    prediction = model.predict(data)
    # print(prediction.shape) # (1, 3)
    a,b = prediction[0]
    print(a,b)
    frame = imutils.resize(frame, width=400)
    # print(type(frame)) # numpy.ndarray
    cv2.imshow('SUN', frame)
    if a>0.999: 
        photo = frame.copy()
        cv2.imwrite(r'C:\Users\littl\Desktop\sticker\photo_'+str(fileNum)+'.jpg', photo)
        send = lineNotify(fileNum)
        fileNum = fileNum + 1
        if send == 400:
            print('Line cannot connect')
            break
        time.sleep(10)
    if cv2.waitKey(1) == ord('s'): # s 鍵關閉視窗
        break

cap.release()
cv2.destroyAllWindows()