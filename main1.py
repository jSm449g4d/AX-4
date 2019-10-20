#9th
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Conv2DTranspose,\
ReLU,Softmax,Flatten,Reshape,UpSampling2D,Input,Activation,LayerNormalization
from tqdm import tqdm
import random


from ARutil import mkdiring,rootYrel

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train,x_test= (x_train.astype(np.float32)/ 256,x_train.astype(np.float32)/ 256)

def tf2img(tfs,dir="./",name="",epoch=0,ext=".png"):
    mkdiring(dir)
    if type(tfs)!=np.ndarray:tfs=tfs.numpy()
    tfs=(tfs*256).astype(np.uint8)
    for i in range(tfs.shape[0]):
        cv2.imwrite(rootYrel(dir,name+"_epoch-num_"+str(epoch)+"-"+str(i)+ext),tfs[i])

def tf_ini():#About GPU resources
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
    if len(physical_devices)==0:print("GPU failed!")
    return len(physical_devices)
tf_ini()

class AE(tf.keras.Model):
    def __init__(self,trials={},opt=keras.optimizers.Adam(1e-3)):
        super().__init__()
        self.layer1=[Flatten(),
                     Dense(128,activation="elu"),
                     Dense(32,activation="elu"),
                     Dropout(0.1),
                     Dense(12,activation="sigmoid")
                     ]
        self.layer2=[Dense(32,activation="elu"),
                     Dense(128,activation="elu"),
                     Dropout(0.1),
                     Dense(28*28,activation="sigmoid"),
                     Reshape((28,28))
                     ]
        self.opt=opt
    @tf.function
    def call(self,mod):
        for i in range(len(self.layer1)):mod=self.layer1[i](mod)
        for i in range(len(self.layer2)):mod=self.layer2[i](mod)
        return mod
    @tf.function
    def pred(self,mod):
        for i in range(len(self.layer2)):mod=self.layer2[i](mod)
        return mod

batch=16

def objective(trial):
    model = AE()
    model.build(input_shape=(batch,28,28))
    model.summary()
        
    optimizer =keras.optimizers.Adam(1e-3)
    for epoch in tqdm(range(30000)):
        ii=random.randint(0,x_train.shape[0]-batch)
        with tf.GradientTape() as tape:
            loss=tf.reduce_mean(keras.losses.binary_crossentropy(
                    x_train[ii:ii+batch],model(x_train[ii:ii+batch])))
            gradients = tape.gradient(loss, model.trainable_variables) 
            gradients,_ = tf.clip_by_global_norm(gradients, 15)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                        
            if epoch % 250 == 0:
                iy=random.randint(0,x_test.shape[0]-batch)
                
                loss=tf.reduce_mean(keras.losses.binary_crossentropy(
                    x_test[iy:iy+batch],model(x_test[iy:iy+batch])))
                
                print("epoch:"+str(epoch)+" loss:"+str(float(loss)))
                loss_for_return=float(loss)
                tf2img(model(x_test[iy:iy+batch]),"./output1_p",epoch=epoch)
                tf2img(x_test[iy:iy+batch],"./output1_t",epoch=epoch)
                tf2img(model.pred(np.random.rand(batch,12).astype(np.float32)),"./output2_p",epoch=epoch)
                
    return loss_for_return
    
objective(0)

