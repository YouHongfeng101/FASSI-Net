
from __future__ import print_function

from sklearn.model_selection import train_test_split
import keras
from PIL import Image
from keras.datasets import mnist
from keras.datasets.mnist import load_data
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Flatten, Conv1D, MaxPooling1D, Dense, Activation, \
    Dropout, GlobalMaxPooling1D, AveragePooling2D, ConvLSTM2D, GlobalMaxPooling2D, Recurrent, Reshape, Bidirectional, \
    BatchNormalization, add, concatenate, activations, Permute, multiply, GRU,SimpleRNN
from keras import backend as K
from keras.layers import Embedding, AveragePooling1D
from keras.engine import Layer
from keras.layers import MaxPooling1D
from keras.optimizers import Adam
import numpy as np
from keras import regularizers, Model, Input
import os
import csv
import tensorflow as tf
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


import os

from IndRNN import IndRNN

os.environ["CUDA_VISIBLE_DEVICES"] = '2' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)




x_s = 256
y_s = 256
channel = 3
num_classes = 7


images = []
lables = []
with open('ISIC2018_Task3_Training_GroundTruth.csv', 'r') as f:
    reader = csv.reader(f)
    for k, row in enumerate(reader):
        if k == 0:
            continue
        img_file = r'E:\youhongfeng\data\Med\Medical 256Class\ISIC2018_Task3_Training_Input/' + row[0] + '.jpg'
        if os.path.exists(img_file):
            img = Image.open(img_file)
            img = img.resize((x_s, y_s), Image.ANTIALIAS)
            img = np.array(img)
            images.append(img)
            lables.append(row.index('1') - 1)
            # img.close()
images = np.array(images)
lables = np.array(lables)
lables = np.reshape(lables,(-1,1))
print(np.array(images).shape)
print(np.array(lables).shape)

'-----------------------------------------------------------------------------------'
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim1 = int(inputs.shape[1])
    input_dim2 = int(inputs.shape[2])
    input_dim3 = int(inputs.shape[3])

    a = Permute((3, 1,2))(inputs)
    a = Reshape((input_dim3, input_dim2,input_dim1))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim2, activation='softmax')(a)

    a_probs = Permute((3,2,1))(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return a_probs


def attention_3d_block_zong(inputs2):
    input_dim1 = int(inputs2.shape[1])
    input_dim2 = int(inputs2.shape[2])
    input_dim3 = int(inputs2.shape[3])

    a = Permute((3, 2,1))(inputs2)
    a = Reshape((input_dim3, input_dim2,input_dim1 ))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim2, activation='softmax')(a)

    b_probs = Permute((3,2,1))(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return b_probs

class D_Att(Layer):

    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = activations.get(activation)
        super(D_Att, self).__init__(**kwargs)


    def call(self, x):
        assert isinstance(x, list)
        X, A1,A2 = x
        A = A1 + A2 + X

        # concatenate2 = K.concatenate([A, X], axis=3)
        return A
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        image_size = input_shape[0][1]
        input_dim = (input_shape[0][0],image_size,image_size, 1 * self.units )
        return input_dim


class ED_Layer(Layer):

    def __init__(self, units,Thr, activation=None, **kwargs):
        self.units = units
        self.Thr = Thr
        self.activation = activations.get(activation)
        super(ED_Layer, self).__init__(**kwargs)


    def call(self, x):
        assert isinstance(x, list)
        A1,A2 ,B1,B2 ,X1,X2= x
        Out = (tf.multiply((A1 - B1), (A1 - B1)) + tf.multiply((A2 - B2), (A2 - B2)))
        Out = K.sqrt(Out)
        Zeos = tf.zeros_like(A1)
        Ones = tf.ones_like(A1)
        print(self.Thr)
        Y = tf.where(Out <self.Thr, x=Ones, y=Zeos)

        Y1 = tf.where(Out > self.Thr, x=Ones, y=Zeos)

        concatenate2 = K.concatenate([Y * X1,  Y * X2], axis=3)
        concatenate3 = K.concatenate([Y1 * X1, Y1 * X2], axis=3)

        return [concatenate2,concatenate3]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        image_size = input_shape[0][1]
        input_dim = (input_shape[0][0],image_size,image_size, 2 * self.units )
        return [input_dim,input_dim]



'-----------------------------------------------------------------------------------'



(x_train, x_test,y_train,y_test)=train_test_split(images,lables,train_size=0.7,stratify=lables,shuffle=True,random_state=1)

(x_val, x_test,y_val,y_test)=train_test_split(x_test,y_test,test_size =0.66,stratify=y_test,shuffle=True,random_state=1)



print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

y_train=np_utils.to_categorical(y_train, num_classes=7)
y_test=np_utils.to_categorical(y_test, num_classes=7)

thr = 0.3
thr1 = 0.3
adam = Adam(lr=0.0001)



model1_7 = Input(shape=(x_s, y_s,channel,))

x = Conv2D(filters=128, kernel_size=7, activation='relu',strides=2)(model1_7)
x = MaxPooling2D(3,strides=2)(x)
x = BatchNormalization()(x)

x = Conv2D(filters=128, kernel_size=5, activation='relu',strides=2)(x)
x = MaxPooling2D(3,strides=2)(x)
x = BatchNormalization()(x)

x = Conv2D(filters=512, kernel_size=3, activation='relu',strides=1)(x)
x = MaxPooling2D(3,strides=2)(x)
xxxx1 = BatchNormalization()(x)

'----------------------------------------------------------------------------------------------------------------------'


x1 = Conv2D(filters=256, kernel_size=2, activation='relu',strides=1,padding='same')(xxxx1)
x11 = attention_3d_block(x1)
x12 = attention_3d_block_zong(x1)
xx1 = D_Att(256)([x1, x11,x12])



x2 = Conv2D(filters=256, kernel_size=3, activation='relu',strides=1,padding='same')(xxxx1)
x21 = attention_3d_block(x2)
x22 = attention_3d_block_zong(x2)
xx2 = D_Att(256)([x2, x21,x22])


x3 = Conv2D(filters=256, kernel_size=4, activation='relu',strides=1,padding='same')(xxxx1)

x31 = attention_3d_block(x3)
x32 = attention_3d_block_zong(x3)
xx3 = D_Att(256)([x3, x31,x32])



x1,Ox1 = ED_Layer(256,thr1)([x11,x12,x21,x22,xx1,xx2])
x1 = Activation('relu')(x1)
x1 = BatchNormalization()(x1)
x2,Ox2 = ED_Layer(256,thr1)([x21,x22,x31,x32,xx2,xx3])
x2 = Activation('relu')(x2)
x2 = BatchNormalization()(x2)
x3,Ox3 = ED_Layer(256,thr1)([x31,x32,x11,x12,xx3,xx1])
x3 = Activation('relu')(x3)
x3 = BatchNormalization()(x3)


x =concatenate([x1,x2,x3])
xxx1 = Conv2D(filters=256, kernel_size=3, activation='relu',strides=1)(x)


print(Ox1)
Oxxx1 = concatenate([Ox1,Ox2,Ox3])
Oxxx1 = MaxPooling2D(2)(Oxxx1)




x1 = Conv2D(filters=256, kernel_size=3, activation='relu',strides=1,padding='same',dilation_rate=2)(xxxx1)
x11 = attention_3d_block(x1)
x12 = attention_3d_block_zong(x1)
xx1 = D_Att(256)([x1, x11,x12])
# xx1 = Conv2D(filters=256, kernel_size=1, activation='relu',strides=1)(xx1)


x2 = Conv2D(filters=256, kernel_size=3, activation='relu',strides=1,padding='same',dilation_rate=3)(xxxx1)
x21 = attention_3d_block(x2)
x22 = attention_3d_block_zong(x2)
xx2 = D_Att(256)([x2, x21,x22])


x3 = Conv2D(filters=256, kernel_size=2, activation='relu',strides=1,padding='same',dilation_rate=2)(xxxx1)

x31 = attention_3d_block(x3)
x32 = attention_3d_block_zong(x3)
xx3 = D_Att(256)([x3, x31,x32])


x1,Ox1 = ED_Layer(256,thr)([x11,x12,x21,x22,xx1,xx2])
x1 = Activation('relu')(x1)
x1 = BatchNormalization()(x1)
x2,Ox2 = ED_Layer(256,thr)([x21,x22,x31,x32,xx2,xx3])
x2 = Activation('relu')(x2)
x2 = BatchNormalization()(x2)
x3,Ox3 = ED_Layer(256,thr)([x31,x32,x11,x12,xx3,xx1])
x3 = Activation('relu')(x3)
x3 = BatchNormalization()(x3)



x =concatenate([x1,x2,x3])
xxx2 = Conv2D(filters=256, kernel_size=3, activation='relu',strides=1)(x)
Oxxx2 = concatenate([Ox1,Ox2,Ox3])
Oxxx2 = MaxPooling2D(2)(Oxxx2)

'-----------------------------------------------------------------------------------------------------------------------'

print(Oxxx1)
O1 = add([Oxxx1,Oxxx2])
O1 = Reshape((-1,256))(O1)





x22 = Conv1D(filters=512, kernel_size=3,strides=1,activation='relu')(O1)
x22 = BatchNormalization()(x22)
shorcut = Conv1D(filters=512, kernel_size=3,strides=1,activation='relu', padding='same')(x22)
shorcut = BatchNormalization()(shorcut)
x = add([x22,shorcut])
x2 = MaxPooling1D(2)(x)


x22 = Conv1D(filters=512, kernel_size=3,strides=1,activation='relu')(x2)
x22 = BatchNormalization()(x22)
shorcut = Conv1D(filters=512, kernel_size=3,strides=1,activation='relu', padding='same')(x22)
shorcut = BatchNormalization()(shorcut)
x = add([x22,shorcut])
x2 = MaxPooling1D(2)(x)



O1 = Bidirectional(LSTM(16,return_sequences=True))(x2)




xxxx2 = concatenate([xxx1,xxx2])

xxxx2 = Conv2D(filters=256, kernel_size=1, activation='relu',strides=1,padding='same')(xxxx2)
xxxx2 = Conv2D(filters=256, kernel_size=3, activation='relu',strides=1,padding='same')(xxxx2)




x = Reshape((-1,32))(xxxx2)
x = concatenate([x,O1],axis=1)
x = Conv1D(filters=256, kernel_size=3, activation='relu',strides=2)(x)
x = Conv1D(filters=256, kernel_size=3, activation='relu',strides=2)(x)
x = MaxPooling1D(3,strides=2)(x)
x = Dropout(0.5)(x)




x = Bidirectional(LSTM(128,return_sequences=False))(x)


all1_output = Dense(num_classes)(x)
all1_output = Activation('softmax')(all1_output)


model = Model(inputs=[model1_7], outputs=[all1_output])
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy']) #categorical_crossentropy

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath="ISIC_Ours.h5",#(就是你准备存放最好模型的地方),
                             monitor='val_acc',#(或者换成你想监视的值,比如acc,loss, val_loss,其他值应该也可以,还没有试),
                             verbose=1,#(如果你喜欢进度条,那就选1,如果喜欢清爽的就选0,verbose=冗余的),
                             save_best_only='True',#(只保存最好的模型,也可以都保存),
                             save_weights_only='True',
                             mode='max',#(如果监视器monitor选val_acc, mode就选'max',如果monitor选acc,mode也可以选'max',如果monitor选loss,mode就选'min'),一般情况下选'auto',
                        period=1)#(checkpoints之间间隔的epoch数)
#损失不下降，则自动降低学习率
lrreduce=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

import time
fit_start = time.clock()
history= model.fit(x_train,y_train,batch_size=16,epochs=100,   verbose=2, shuffle =True,validation_data = (x_val, y_val),callbacks = [checkpoint])
fit_end = time.clock()



model.load_weights('ISIC_Ours.h5')

t_start = time.clock()
loss,acc = model.evaluate(x_test,y_test,verbose=2)
t_end = time.clock()
print('Test loss :',loss)
print('Test accuracy :',acc)
print("test time is: ",t_end-t_start)
aaa = 0

y_pred_class = model.predict(x_test)

list3 = np.argmax(y_pred_class,axis=1)
list22 = np.argmax(y_test,axis=1)

from sklearn import metrics


def kappa(confusion_matrix, k):
    dataMat = np.mat(confusion_matrix)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe  = float(ysum*xsum)/np.sum(dataMat)**2
    OA = float(P0/np.sum(dataMat)*1.0)
    cohens_coefficient = float((OA-Pe)/(1-Pe))
    return cohens_coefficient

classify_report = metrics.classification_report(list22, list3)
confusion_matrix = metrics.confusion_matrix(list22, list3)
f1 = metrics.f1_score(list22, list3, average='macro')

overall_accuracy = metrics.accuracy_score(list22, list3)
acc_for_each_class = metrics.precision_score(list22, list3, average=None)
average_accuracy = np.mean(acc_for_each_class)
kappa_coefficient = kappa(confusion_matrix, 7)
print('classify_report : \n', classify_report)
print('confusion_matrix : \n', confusion_matrix)
print('acc_for_each_class : \n', acc_for_each_class)
print('average_accuracy: {0:f}'.format(average_accuracy))
print('overall_accuracy: {0:f}'.format(overall_accuracy))
print('kappa coefficient: {0:f}'.format(kappa_coefficient))
print('F1: ',f1)
newpath = r'E:\youhongfeng\End-to-End\在分类\实验结果\ISIC_Ours.txt'
f = open(newpath,'w')
f.write(classify_report)
# f.write(confusion_matrix)
f.write(str(acc_for_each_class.tolist()))
f.write('\n')
f.write('average_accuracy:{0:f}'.format(average_accuracy))
f.write('\n')
f.write('overall_accuracy:{0:f}'.format(overall_accuracy))
f.write('\n')
f.write('kappa coefficient:{0:f}'.format(kappa_coefficient))
f.write('\n')
f.write('f1:{0:f}'.format(f1))
f.write('\n')
f.close()

x_tick=['MEL','NV','BCC','AKIEC','BKL','DF','VASC']
y_tick=['MEL','NV','BCC','AKIEC','BKL','DF','VASC']

import pandas as pd


con_mat_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=0)
con_mat_norm=pd.DataFrame(con_mat_norm,index=y_tick,columns=x_tick)
print(con_mat_norm.shape)
con_mat_norm = np.around(con_mat_norm, decimals=2)

plt.figure(figsize=(10, 10))
sns.set(font_scale=2.4)
sns.heatmap(con_mat_norm, annot=True, cmap='Blues')

plt.title("ISIC: ISIC_Ours")
plt.ylim(0, 7)
plt.xlabel('PL')
plt.ylabel('TL')
plt.savefig(r"E:\youhongfeng\End-to-End\在分类\实验结果/ISIC_Ours.jpg")
plt.show()
