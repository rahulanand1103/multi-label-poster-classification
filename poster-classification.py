#library
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout,Conv2D,MaxPool2D,BatchNormalization,AveragePooling2D
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

#read csv file
df=pd.read_csv("Movies-Poster_Dataset/train.csv")

#image preprocessing
img_width=250
img_height=250

x=[]

for i in tqdm(range(df.shape[0])):
  path="/content/Movies-Poster_Dataset/Images/"+df['Id'][i]+'.jpg'
  img=image.load_img(path,target_size=(img_width,img_height,3))
  img=image.img_to_array(img)
  img=img/255.0
  x.append(img)

x=np.array(x)
y=df.drop(['Id','Genre'],axis=1)
y=y.to_numpy()
y.shape


#train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


#Deep Learning model
model=Sequential()

model.add(Conv2D(16,(3,3),activation='relu',input_shape=X_train[0].shape))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Dense(25,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10)
