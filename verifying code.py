# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:47:33 2019

@author: user
"""

import numpy as np
import keras
import keras.backend as K
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization, Activation, PReLU
from keras import regularizers
from keras.datasets import mnist
from keras.layers import Convolution2D as Conv2D
import pandas as pd
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.models import Sequential
from scipy import io
import scipy
from sklearn.model_selection import train_test_split
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D, Dense, Flatten, Reshape
from keras.optimizers import SGD
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os   
import cv2

tf.reset_default_graph()

dataset = scipy.io.loadmat('train.mat')
x_orig=np.array(dataset["x_train"][:])
y_orig=np.array(dataset["y_train"][:])
#X_train,X_test,y_train_label,y_test_label=train_test_split(x_orig,y_orig,test_size=0,random_state=0)


extra = scipy.io.loadmat('midterm_test.mat')
extra_X_test=np.array(extra["x_test"][:])
extra_y_test_label=np.array(extra["y_test"][:])
#extra_X_test,extra_X_train,extra_y_test_label,extra_y_train_label=train_test_split(extra_x_orig,extra_y_orig,test_size=1.0,random_state=0)



print(extra_X_test.shape)

print(extra_y_test_label.shape)

#print(X_train.shape)
#print(X_test.shape)
#print(y_train_label.shape)
#print(y_test_label.shape)
from keras.utils.np_utils import to_categorical
X_train = x_orig.astype('float32')
extra_X_test = extra_X_test.astype('float32')
X_train /= 255.0
extra_X_test /= 255.0
X_train = np.reshape(X_train, [-1,48,170,1])
extra_X_test = np.reshape(extra_X_test, [-1,48,170,1])


y_train = to_categorical(y_orig)
extra_y_test= to_categorical(extra_y_test_label)


x_train_new=np.empty([5838,48,108,1],dtype=float)
extra_X_test_new=np.empty([2000,48,108,1],dtype=float)
x_train_1=np.empty([5838,48,27,1],dtype=float)
x_train_2=np.empty([5838,48,27,1],dtype=float)
x_train_3=np.empty([5838,48,27,1],dtype=float)
x_train_4=np.empty([5838,48,27,1],dtype=float)
extra_X_test_1=np.empty([2000,48,27,1],dtype=float)
extra_X_test_2=np.empty([2000,48,27,1],dtype=float)
extra_X_test_3=np.empty([2000,48,27,1],dtype=float)
extra_X_test_4=np.empty([2000,48,27,1],dtype=float)

for i in range(5838):
    for j in range(48):
        x_train_new[i,j]=X_train[i,j,14:122:]

for i in range(2000):
    for j in range(48):
        extra_X_test_new[i,j]=extra_X_test[i,j,14:122:] 
        
        
for i in range(5838):
    for j in range(48):
        x_train_1[i,j]=x_train_new[i,j,0:27:]    
        
        
for i in range(5838):
    for j in range(48):
        x_train_2[i,j]=x_train_new[i,j,27:54:] 
        
        
for i in range(5838):
    for j in range(48):
        x_train_3[i,j]=x_train_new[i,j,54:81:]   
                
for i in range(5838):
    for j in range(48):
        x_train_4[i,j]=x_train_new[i,j,81:108:] 
        
for i in range(2000):
    for j in range(48):
        extra_X_test_1[i,j]=extra_X_test_new[i,j,0:27:]    
        
        
for i in range(2000):
    for j in range(48):
        extra_X_test_2[i,j]=extra_X_test_new[i,j,27:54:] 
        
        
for i in range(2000):
    for j in range(48):
        extra_X_test_3[i,j]=extra_X_test_new[i,j,54:81:]   
                
for i in range(2000):
    for j in range(48):
        extra_X_test_4[i,j]=extra_X_test_new[i,j,81:108:] 
        
      
y_train_1=np.empty([5838,19],dtype=int)   
y_train_2=np.empty([5838,19],dtype=int) 
y_train_3=np.empty([5838,19],dtype=int) 
y_train_4=np.empty([5838,19],dtype=int)   
extra_y_test_1=np.empty([2000,19],dtype=int)   
extra_y_test_2=np.empty([2000,19],dtype=int) 
extra_y_test_3=np.empty([2000,19],dtype=int) 
extra_y_test_4=np.empty([2000,19],dtype=int)     
        
for i in range(5838):
    y_train_1[i]=y_train[i,0, :]         

for i in range(5838):
    y_train_2[i]=y_train[i,1, :]
    
for i in range(5838):
    y_train_3[i]=y_train[i,2, :]

for i in range(5838):
    y_train_4[i]=y_train[i,3, :]
    
for i in range(2000):
    extra_y_test_1[i]=extra_y_test[i,0, :]         

for i in range(2000):
    extra_y_test_2[i]=extra_y_test[i,1, :]
    
for i in range(2000):
    extra_y_test_3[i]=extra_y_test[i,2, :]

for i in range(2000):
    extra_y_test_4[i]=extra_y_test[i,3, :]



        
        
XXX_train=np.vstack((x_train_1,x_train_2,x_train_3,x_train_4))                
YYY_train=np.vstack((y_train_1,y_train_2,y_train_3,y_train_4)) 
extra_XXX_test=np.vstack((extra_X_test_1,extra_X_test_2,extra_X_test_3,extra_X_test_4))        
extra_YYY_test=np.vstack((extra_y_test_1,extra_y_test_2,extra_y_test_3,extra_y_test_4))        

for t in range(5):
        pic1 = extra_X_test_1[t]
        plt.subplot(5,4,(1+(t*4)))
        pic1 = pic1.reshape((48,27))
        plt.imshow(pic1,cmap='gray')
        pic2 = extra_X_test_2[t]
        pic2 = pic2.reshape((48,27 ))
        plt.subplot(5,4,(2+(t*4)))
        plt.imshow(pic2,cmap='gray')
        pic3 = extra_X_test_3[t]
        pic3 = pic3.reshape((48,27 ))
        plt.subplot(5,4,(3+(t*4)))
        plt.imshow(pic3,cmap='gray')
        pic4 = extra_X_test_4[t]
        pic4 = pic4.reshape((48,27 ))
        plt.subplot(5,4,(4+(t*4)))
        plt.imshow(pic4,cmap='gray')  
'''pic1 = XXX_test[0:3]
# 图片分辨率是40*40
pic1 = pic1.reshape((48,30 ))
# 绘图
from matplotlib import pyplot as plt
# 传入cmap='gray'指定图片为黑白
plt.imshow(pic1, cmap='gray')'''


#print(y_train_1.shape)  
#print(y_train_1)        
#print(x_train_1.shape)
#print(XXX_train.shape)
#print(YYY_train.shape)
#print(YYY_test.shape)
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_train_hot.shape)
#print(y_train[6,3])
#print(x_train_new.shape)
#print(x_test_new.shape)
#print(y_train[0,0].shape)



import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D, Dense, Flatten, Reshape
from keras.optimizers import SGD
tf.reset_default_graph()
keras.backend.clear_session()
n_filters=[32,64]
n_classes = 19  # 0-9 digits
n_width = 48
n_height = 27
n_depth = 1
n_inputs = n_height * n_width * n_depth  # total pixels
learning_rate = 0.03
n_epochs = 25
batch_size = 300
model = Sequential()
model.add(Dense(1, input_shape=(n_width,n_height,n_depth)))
model.add(Conv2D(filters=n_filters[0], 
                 kernel_size=4, 
                 padding='SAME', 
                 activation='relu' 
                ) 
         )
model.add(MaxPooling2D(pool_size=(2,2), 
                       strides=(2,2) 
                      ) 
         )
model.add(Conv2D(filters=n_filters[1], 
                 kernel_size=4, 
                 padding='SAME', 
                 activation='relu', 
                ) 
         )
model.add(MaxPooling2D(pool_size=(2,2), 
                       strides=(2,2) 
                      ) 
         )
model.add(Conv2D(filters=n_filters[0], 
                 kernel_size=4, 
                 padding='SAME', 
                 activation='relu' 
                ) 
         )
model.add(MaxPooling2D(pool_size=(2,2), 
                       strides=(2,2) 
                      ) 
         )
model.add(Flatten())
model.add(Dense(units=1024, activation='relu'))
model.add(Dense(units=n_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=learning_rate),
              metrics=['accuracy'])
model.fit(XXX_train, YYY_train,
                    batch_size=batch_size,
                    epochs=n_epochs)

score = model.evaluate(extra_XXX_test, extra_YYY_test)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])




 
list_LABEL=[2,3,4,5,7,9,'A','C','F','H','K','M','N','P','Q','R','T','Y','Z']

def one_word_test(extra_XXX_test,extra_YYY_test,num,total):
    #count=defaultdict(int)
    t=''
    prediction= model.predict_classes(extra_XXX_test)
    count=0
    for i in range(num):
        for j in range(19):
            if extra_YYY_test[i,j]==1:
                
                t=list_LABEL[j]
                
        if list_LABEL[prediction[i]]==t:
            count += 1
        else:
            count=count
    return count
    

 
def FINAL_CALL(extra_XXX_test,extra_YYY_test,num,total_num):    #total_num是四個字合起來的圖的總張數 
    prediction= model.predict_classes(extra_XXX_test)     #num是要跑幾張
    Scount=0
    Total_count=0
    y_label = []
    x_label = []
    W=0
    for i in range(num) :         
        for m in range(5):
            if m<4:                
                for j in range(19):
                     if extra_YYY_test[(i+(m*(total_num))),j]==1:                 
                         t=list_LABEL[j]
                x_label.append (list_LABEL[prediction[i+(m*(total_num))]])                
                y_label.append(t)                             
            else:                
                #print("predict="+ str(list_LABEL[prediction[i]]))
                print("Prediction="+str(x_label))
                print("Answer="+str(y_label))
                for k in range(4):                
                    if x_label[k]==y_label[k]:
                        Scount=Scount
                    else:
                        Scount +=1
                print("wrongword="+str(Scount))
                if(Scount==0):
                    Total_count+=1
                else:
                    Total_count=Total_count
                    Scount=0
                y_label.clear()
                x_label.clear()
                                                             
    print("Total accuracy="+str(Total_count))
    print("Total test="+str(num))
    print("Total accuracy rate="+str(Total_count/(num)))
            

 
'''FINAL_CALL(extra_XXX_test,extra_YYY_test,2000,2000)'''
print("amount of wrong words="+str(8000-one_word_test(extra_XXX_test,extra_YYY_test,8000,8000))) 
print("one word accuracy="+str(one_word_test(extra_XXX_test,extra_YYY_test,8000,8000)))  
print("one word accuracy rate="+str(one_word_test(extra_XXX_test,extra_YYY_test,8000,8000)/8000)) 




'''
prediction= model.predict_classes(XXX_test)
print(list_LABEL[prediction[1]])
pic3 = x_test_new[1]
pic3 = pic3.reshape((48,108 ))    
plt.imshow(pic3, cmap='gray')'''
'''   
 
def FINAL_CALL(XXX_test,YYY_test,num,total_num):    #total_num是四個字合起來的圖的總張數 
    prediction= model.predict_classes(XXX_test)     #num是要跑幾張
    Scount=0
    Total_count=0
    y_label = []
    x_label = []
    W=0
    for i in range(num) : 
        print('n')
        for m in range(4):
            for j in range(19):
                 if YYY_test[(i+(m*(total_num))),j]==1:                 
                     t=list_LABEL[j]
            #if len(x_label)<4 && i==0 && m==3 
            if len(x_label)<4: 
                
                x_label.append (list_LABEL[prediction[i+(m*(total_num))]])                
                y_label.append(t)                
            else:
                print('m')
                #print("predict="+ str(list_LABEL[prediction[i]]))
                print("Prediction="+str(x_label))
                print("Answer="+str(y_label))
                for k in range(4):                
                    if x_label[k]==y_label[k]:
                        Scount=Scount
                    else:
                        Scount +=1
                print("wrongword="+str(Scount))
                if(Scount==0):
                    Total_count+=1
                else:
                    Total_count=Total_count
                    Scount=0
                y_label.clear()
                x_label.clear()
                x_label.append (list_LABEL[prediction[i+(m*(total_num))]])
                y_label.append(t)                                                       
    print("Total accuracy="+str(Total_count))
    print("Total test="+str(num))
    print("Total accuracy rate="+str(Total_count/(num)))
            

 
FINAL_CALL(XXX_test,YYY_test,1,1168)
print("one word accuracy="+str(one_word_test(XXX_test,YYY_test,4672,4672)))  
print("one word accuracy="+str(one_word_test(XXX_test,YYY_test,4672,4672)/4672)) 
         
                
'''

from matplotlib import pyplot as plt

'''
pic1 = x_test_new[3]
pic1 = pic1.reshape((48,108 ))    
plt.imshow(pic1, cmap='gray')

pic2 = x_test_new[1]
pic2 = pic2.reshape((48,108 ))    
plt.imshow(pic2, cmap='gray')

pic3 = x_test_new[3]
pic3 = pic3.reshape((48,108 ))    
plt.imshow(pic3, cmap='gray')
'''



'''

prediction= model.predict_classes(XXX_test)
print("predict="+ str(list_LABEL[prediction[1168]]))'''






'''for i in range(15) :
    print(prediction[i])     '''    
#prediction= model.predict(XXX_test)  



'''def four_word_test(YYY_test,num):
    #count=defaultdict(int)
    t=''
    count=0
    c=0
    Scount=0
    for i in range(num):
        for j in range(19):
            if YYY_test[i,j]==1:
                y_label.append(list_LABEL[j]) 
                t=list_LABEL[j]
        if c<=3:
            if list_LABEL[prediction[i]]==t:
                Scount +=1
            c+=1
        else:
            c=0
            print("rightword="+str(Scount))
            print("wrongword="+str(4-Scount))
            Scount=0
      
four_word_test(YYY_test,400)        '''
     
'''y_Trainl = np.reshape(y_train,(-1,1))
y_trans = []'''

#print(YYY_test[0,18])

#print (list_LABEL[9])
#y_label = []
'''def label_transfer(YYY_test,num):
    for i in range(10):
        for j in range(19):
            if YYY_test[i,j]==1:
                y_label.append(list_LABEL[j])    
                #print (list_LABEL[j])
                
label_transfer(YYY_test,10)
print (y_label) 

        ''' 
