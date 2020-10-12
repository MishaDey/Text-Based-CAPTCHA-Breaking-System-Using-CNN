import keras
import random
import numpy as np
import tensorflow as tf 
import tensorflow.compat.v1.keras.backend as Keras_backend
from keras.utils import Sequence
from captcha.image import ImageCaptcha
#configure the session to prevent tensorflow from occupying all the video memmory
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth=True
session= tf.compat.v1.Session(config=config)
Keras_backend.set_session(session)
 
class Captcha_Sequence_Generator(Sequence):
    #initialization function of the class
    def __init__(self,characters,batch_size,steps,n_len=5,width=128,height=64):
        self.characters=characters
        self.batch_size=batch_size
        self.steps=steps
        self.n_len=n_len #length of string
        self.width=width
        self.height=height
        self.n_class=len(characters) #character code species = 26(lower case letters) + 26(uppercase letter) + 10(0-9 digits)
        self.generator=ImageCaptcha(width=width,height=height)
    
    def __getitem__(self,idx):
        #creating empty input and output values
        X_val=np.zeros((self.batch_size,self.height,self.width,3),dtype=np.float32) # 3 because we are using a RGB image 
        Y_val=[np.zeros((self.batch_size,self.n_class),dtype=np.uint8) for ind_1 in range(self.n_len)]
        #generate items for each batch size
        for i in range(self.batch_size):
            Captcha_String=''.join([random.choice(self.characters) for ind_2 in range(self.n_len)])
            Captcha = self.generator.generate_image(Captcha_String)
            X_val[i] = np.array(Captcha)/255.0
            #multi-hot encoding to get the Y values
            for ind_3,char in enumerate(Captcha_String):
                Y_val[ind_3][i,:]=0 
                Y_val[ind_3][i,self.characters.find(char)]=1
        return X_val,Y_val
    
    def __len__(self):
        return self.steps 
    