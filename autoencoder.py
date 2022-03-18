import numpy as np
import pandas as pd
import pathlib
import cv2
import os
import glob
import tensorflow as tf
dir_path ="C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/back_pack"
data_root = pathlib.Path(dir_path)

def return_file_names(root_dir):
    arr = os.listdir(root_dir)
    
    #col=[]
    li=[]
    for i in arr:
         dir_path = str(root_dir)+'\\'+ i
        
         li.append(dir_path)
         
         
         
    data_df = pd.DataFrame()
    data_df['path_img']=li
    return data_df
df_train = return_file_names(data_root)

dir_path2 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/bike"
data_root2 = pathlib.Path(dir_path2)
df_train2 = return_file_names(data_root2)

dir_path3 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/bike_helmet"
data_root3 = pathlib.Path(dir_path3)
df_train3 = return_file_names(data_root3)

dir_path4 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/bookcase"
data_root4 = pathlib.Path(dir_path4)
df_train4 = return_file_names(data_root4)

dir_path5 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/bottle"
data_root5 = pathlib.Path(dir_path5)
df_train5 = return_file_names(data_root5)

dir_path6 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/calculator"
data_root6 = pathlib.Path(dir_path6)
df_train6 = return_file_names(data_root6)

dir_path7 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/desk_chair"
data_root7 = pathlib.Path(dir_path7)
df_train7 = return_file_names(data_root7)

dir_path8 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/desk_lamp"
data_root8 = pathlib.Path(dir_path8)
df_train8 = return_file_names(data_root8)

dir_path9 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/desktop_computer"
data_root9 = pathlib.Path(dir_path9)
df_train9 = return_file_names(data_root9)

dir_path10 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/file_cabinet"
data_root10 = pathlib.Path(dir_path10)
df_train10= return_file_names(data_root10)

dir_path11 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/headphones"
data_root11 = pathlib.Path(dir_path11)
df_train11 = return_file_names(data_root11)

dir_path12 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/keyboard"
data_root12 = pathlib.Path(dir_path12)
df_train12= return_file_names(data_root12)

dir_path13 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/laptop_computer"
data_root13 = pathlib.Path(dir_path13)
df_train13 = return_file_names(data_root13)

dir_path14 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/letter_tray"
data_root14 = pathlib.Path(dir_path14)
df_train14 = return_file_names(data_root14)

dir_path15 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/mobile_phone"
data_root15 = pathlib.Path(dir_path15)
df_train15 = return_file_names(data_root15)

dir_path16 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/monitor"
data_root16 = pathlib.Path(dir_path16)
df_train16= return_file_names(data_root16)

dir_path17= "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/mouse"
data_root17= pathlib.Path(dir_path17)
df_train17= return_file_names(data_root17)

dir_path18= "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/mug"
data_root18 = pathlib.Path(dir_path18)
df_train18= return_file_names(data_root18)

dir_path19 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/paper_notebook"
data_root19 = pathlib.Path(dir_path19)
df_train19= return_file_names(data_root19)

dir_path20 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/pen"
data_root20 = pathlib.Path(dir_path20)
df_train20= return_file_names(data_root20)

dir_path21= "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/phone"
data_root21= pathlib.Path(dir_path21)
df_train21= return_file_names(data_root21)

dir_path22= "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/printer"
data_root22= pathlib.Path(dir_path22)
df_train22= return_file_names(data_root22)

dir_path23 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/projector"
data_root23 = pathlib.Path(dir_path23)
df_train23 = return_file_names(data_root23)



dir_path25= "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/punchers"
data_root25= pathlib.Path(dir_path25)
df_train25= return_file_names(data_root25)

dir_path26= "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/ring_binder"
data_root26= pathlib.Path(dir_path26)
df_train26= return_file_names(data_root26)

dir_path27= "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/ruler"
data_root27= pathlib.Path(dir_path27)
df_train27= return_file_names(data_root27)

dir_path28= "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/scissors"
data_root28= pathlib.Path(dir_path28)
df_train28= return_file_names(data_root28)

dir_path29= "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/speaker"
data_root29 = pathlib.Path(dir_path29)
df_train29= return_file_names(data_root29)

dir_path30 = "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/stapler"
data_root30 = pathlib.Path(dir_path30)
df_train30= return_file_names(data_root30)

dir_path31= "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/tape_dispenser"
data_root31= pathlib.Path(dir_path31)
df_train31= return_file_names(data_root31)

dir_path32= "C:/Users/shiva/Desktop/Python/Machine Learning/autoencoders/archive/Office31/amazon/trash_can"
data_root32= pathlib.Path(dir_path32)
df_train32= return_file_names(data_root32)

frames = [df_train,df_train2,df_train3,df_train4,df_train5,df_train6,df_train7,df_train8,df_train9,df_train10,df_train11,df_train12,df_train13,df_train14,df_train15,df_train16,df_train17,df_train18,df_train19,df_train20,df_train21,df_train22,df_train23,df_train25,df_train26,df_train27,df_train28,df_train29,df_train30,df_train31,df_train32]
df_merge = pd.concat(frames,ignore_index=True)

from matplotlib.pyplot import imshow
import numpy as np
import cv2

from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
import os
from tqdm import tqdm
SIZE=256

def gaussiannoise(img):
    gauss = np.random.normal(0,1,img.size)
    gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
    img=cv2.add(img,gauss)
    return img

import matplotlib.pyplot as plt 
filepath = df_merge['path_img'][0]
img = cv2.imread(filepath,1)
plt.imshow(img)
plt.show()

img = gaussiannoise(img)
plt.imshow(img)
plt.show()


li=[]
for i in range(0,2817):
    img_data=[]
    filepath = df_merge['path_img'][i]
    img = cv2.imread(filepath,1)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = gaussiannoise(img)
    img=cv2.resize(img,(SIZE, SIZE))
    img_data.append(img_to_array(img))
    img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE,3))
    img_array = img_array.astype('float32') / 255.
    li.append(img_array)
    
    #np.concatenate(li)
li=np.array(li)    

from sklearn.model_selection import train_test_split
X_train, X_test= train_test_split(li,test_size=0.20, random_state=42) 

X_train = X_train.reshape(X_train.shape[0], 256, 256,3)
X_test = X_test.reshape(X_test.shape[0], 256, 256,3)
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.

#model = Sequential()
#model.add(Conv2D(28, (3, 3), activation='relu', padding='same', input_shape=(28,28, 3)))
#model.add(MaxPooling2D((2, 2), padding='same'))
#model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D((2, 2), padding='same'))
#model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

#model.add(MaxPooling2D((2, 2), padding='same'))

#model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
#model.add(UpSampling2D((2, 2)))
#model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
#model.add(UpSampling2D((2, 2)))
#model.add(Conv2D(28, (3, 3), activation='relu', padding='same'))
#model.add(UpSampling2D((2, 2)))
#model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

#Convolutional autoencoder
# Encoder
from keras.layers import Input, add
x = Input(shape=(256,256,3)) 
conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)
conv1_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D((2, 2), padding='same')(conv1_2)
conv1_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2)
pool3 = MaxPooling2D((2, 2), padding='same')(conv1_3)
conv1_4 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool3)
pool4 = MaxPooling2D((2, 2), padding='same')(conv1_4)
conv1_5 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool4)
pool5 = MaxPooling2D((2, 2), padding='same')(conv1_5)
conv1_6 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool5)
h = MaxPooling2D((2, 2), padding='same')(conv1_6)




# Decoder
conv2_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
up1 = UpSampling2D((2, 2))(conv2_1)
conv2_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(up1)
up2 = UpSampling2D((2, 2))(conv2_2)
conv2_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(up2)
up3 = UpSampling2D((2, 2))(conv2_3)
conv2_4 = Conv2D(8, (3, 3), activation='relu', padding='same')(up3)
up4 = UpSampling2D((2, 2))(conv2_4)
conv2_5 = Conv2D(8, (3, 3), activation='relu', padding='same')(up4)
up5 = UpSampling2D((2, 2))(conv2_5)
conv2_6 = Conv2D(32, (3, 3), activation='relu',padding='same')(up5)
up6 = UpSampling2D((2, 2))(conv2_6)
r = Conv2D(3, (3, 3), activation='softmax', padding='same')(up6)
from keras.models import Model
autoencoder = Model(inputs=x, outputs=r)
autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])

#model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
autoencoder.summary()

epochs = 50
batch_size = 16

history = autoencoder.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, X_test))

#model.fit(X_train,X_train,
 #       epochs=50,
  #      batch_size=256,
   #     validation_data=(X_test,X_test),
    #    verbose=1,
     #   shuffle=True) 
     
decoded_imgs = autoencoder.predict(X_test) 
import matplotlib.pyplot as plt 
from PIL import Image
n = 3
plt.figure(figsize=(20, 6))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i+1)
    img = Image.fromarray(X_test[i].reshape(256,256,3),'RGB')
    #img.show()
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    
    # display reconstruction 
    ax = plt.subplot(3, n, i+n+1)
    plt.imshow(decoded_imgs[i].reshape(256, 256,3))
    plt.gray()
    img2 = Image.fromarray(decoded_imgs[i].reshape(256, 256,3),'RGB')
    #img2.show()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    

#imgplot = plt.imshow(X_test[1])
img = Image.fromarray(X_test[0].reshape(256,256,3),'RGB')
img.show()

plt.imshow(X_test[0], interpolation='nearest')
plt.show()

#for i in range(5):
    #plt.figure()
    #img_name = images[i]
    
    
    #img = cv2.imread(img_name)
    
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.xlabel(captions_dict[img_name.split('\\')[-1]])
    #plt.imshow(img)    
    
    
    