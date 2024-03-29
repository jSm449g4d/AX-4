#9th
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Conv2DTranspose,\
ReLU,Softmax,Flatten,Reshape,UpSampling2D,Input,Activation,LayerNormalization
import random
import optuna
import pickle
import csv
import codecs

from ARutil import mkdiring,rootYrel

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train,x_test= (x_train.astype(np.float32)/ 256,x_test.astype(np.float32)/ 256)

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
    def __init__(self,trial={},opt=keras.optimizers.Adam(1e-3)):
        super().__init__()
        self.layer1=[Flatten(),
                     Dense(trial.suggest_int("a", 32,256),activation="elu"),
                     Dense(trial.suggest_int("b", 32,256),activation="elu"),
                     Dropout(trial.suggest_uniform("A",0,0.5)),
                     Dense(12,activation="sigmoid")
                     ]
        self.layer2=[Dense(trial.suggest_int("c", 32, 256),activation="elu"),
                     Dense(trial.suggest_int("d", 32, 256),activation="elu"),
                     Dropout(trial.suggest_uniform("B",0,0.5)),
                     Dense(28*28,activation="sigmoid"),
                     Reshape((28,28))
                     ]
        self.opt=opt
    def call(self,mod):
        for i in range(len(self.layer1)):mod=self.layer1[i](mod)
        for i in range(len(self.layer2)):mod=self.layer2[i](mod)
        return mod
    def pred(self,mod):
        for i in range(len(self.layer2)):mod=self.layer2[i](mod)
        return mod

batch=32

class K_B(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        iy=random.randint(0,x_test.shape[0]-batch)
        tf2img(self.model(x_test[iy:iy+batch]),"./output1_p",epoch=epoch)
        tf2img(x_test[iy:iy+batch],"./output1_t",epoch=epoch)
        tf2img(self.model.pred(np.random.rand(batch,12).astype(np.float32)),"./output2_p",epoch=epoch)

def objective(trial,FT=0):
    model = AE(trial=trial)
    model.build(input_shape=(batch,28,28))
    model.summary()
    
    callbacks=[]
    if FT:print("*****Best Params is",trial.params);callbacks=[K_B()]
    
    model.compile(optimizer =keras.optimizers.Adam(1e-3),
                          loss=keras.losses.binary_crossentropy,
                          metrics=['accuracy'])
    hist=model.fit(x_train,x_train,batch_size=batch,epochs=10,callbacks=callbacks,validation_split=0.1)
    
    if FT:print(hist.history,file=codecs.open('hist.txt', 'w', 'utf-8'))
    
    if FT==0:
        tm_dict={"loss":(hist.history['val_loss'])[-1]};tm_dict.update(trial.params)
        with open("file.csv","a+",encoding='utf-8') as f:
            writer = csv.DictWriter(f,lineterminator='\n',fieldnames=list(tm_dict.keys()))
            if f.tell()==0:f.write("#");writer.writeheader()
            writer.writerow(tm_dict)
            
    return -(hist.history['val_loss'])[-1]

if __name__ == '__main__':
    try:
        with open('./opt.pickle', 'rb') as f:study = pickle.load(f)
    except:
        study = optuna.create_study();
    study.optimize(objective, n_trials=100)
    with open('./opt.pickle', 'wb') as f:pickle.dump(study, f)
    print(study.best_params,":",study.best_value)
    objective(optuna.trial.FixedTrial(study.best_params),FT=1)
    
