import keras
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.utils import plot_model
import matplotlib.pyplot as plt
from IPython.display import display,Image
from keras.optimizers import Adam
from keras.callbacks import *
from Utils_funX import Decode_Y_Val

def CNN_Model_Initialize(height,width,n_classes,n_len):
    #feature extraction 
    input_tensor=Input((height,width,3))
    out=input_tensor
    # modifing x -- that is modyfying the output tensor
    # No. of Colvolution blocks = 5
    for n_blocks in range(5):
        convl_num=2 #no. of convoulutional Layer=2
        #Building each convolution Layer
        for i in range(convl_num):
            #no. of nodes/filters = 32*2^(0)=32 for 1st Conv LAyer 
            #and 32*2^(1)=64 for the 2nd Conv layer
            # padding = "same" -> input volume size matches the output volume size
            # kernel_initializer = he_uniform as it Draws samples from a uniform distribution within [-limit, limit], where limit = sqrt(6 / fan_in) (fan_in is the number of input units in the weight tensor).
            out=Conv2D(32*2**min(n_blocks,3),kernel_size=3,activation='relu',padding='same',kernel_initializer='he_uniform')(out)
            out=BatchNormalization()(out) #standardizes the inputs to a layer for each mini-batch
        # One Pooling Layer
        out = MaxPooling2D(2)(out)
    out = Flatten()(out)
    #fully-Connected Layer
    out = [Dense(n_classes,activation='softmax')(out) for i in range(n_len)]
    model=Model(inputs=input_tensor,outputs=out) 
    return model

def CNN_model_visualize(model):
    plot_model(model,to_file='CNN_Model_Plot.png',show_shapes=True)
    display(Image('CNN_Model_Plot.png'))

       
def CNN_model_Compile_and_Train(model,Train_data,Test_data,train_num):
    # Patience= 4 ->The no. no of epochs with no improvement - metric is "loss"
    # CsvLogger -> streams the epoch Results to a csv file
    # Adam Optimizer --> Learning Rate Automatically Set
    model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001*(0.1**(float(train_num-1))),amsgrad=True),metrics=['accuracy'])
    call_backs=[EarlyStopping(patience=4),CSVLogger("CNN_Model_Epochs.csv"),ModelCheckpoint('CNN_Model.h5',save_best_only=True)]
    history=model.fit_generator(Train_data,epochs=100,validation_data=Test_data,callbacks=call_backs,workers=5,use_multiprocessing=True)
    return history

def CNN_Model_Test(model,Test_data,characters):
    x_val=[]
    y_val=[]
    y_pred=[]
    for i in range(len(Test_data)):
        x,y=Test_data[i]
        y_p = model.predict(x)
        
        y_char,char_classes=Decode_Y_Val(y,characters)
        y_val.append(char_classes)
        py_char,pchar_classes=Decode_Y_Val(y_p,characters)
        y_pred.append(pchar_classes)
        
        plt.title("Y_TRUE : {}  ,  Y_PRED : {} ".format(y_char,py_char)) 
        plt.imshow(x[0],cmap='gray')
        plt.show()        
    return y_val,y_pred  