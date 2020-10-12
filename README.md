# Text-Based-CAPTCHA-Breaking-System-Using-CNN
A Text-Based CAPTCHA(Completely Automated Public Turing Test) Breaking System Using CNN(Convolutional Neural Networks). Dataset: Generated using the "Captcha" library.

The tern 'CAPTCHA' Stands for Completely Automated Public Turing Test.

## Libraries/Modules Used :

       1. Tensorflow

       2. Scikit-learn

       3. Captcha

       4. Numpy

       5. Pandas

       6. Matplotlib

      7. IPython

## Dataset Description :

The train and test dataset was generated using the "captcha" module/library, used to generate captchas/verification codes for the given set of characters.

## STEPS:

### 1. Captcha Sequence Generator: is used for the real-time data feeding to the CNN model

     Captcha Character Set = Set of all Uppercase Alphabets + Set of all Lowercase Applabets + Set of digits (0-9)

     " __init__ " function used for initialization

     "__len__" return the number of batches per epochs

     "__getitem__"  It is executed when a batch corresponding to a batch index is called to generate it.

### 2. Convolutional Neural Network Model : 

#### Overall Accuracy = 0.95578125




## Model.Summary() 

Model: "functional_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 64, 128, 3)] 0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 64, 128, 32)  896         input_1[0][0]                    
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 64, 128, 32)  128         conv2d[0][0]                     
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 64, 128, 32)  9248        batch_normalization[0][0]        
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 64, 128, 32)  128         conv2d_1[0][0]                   
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 32, 64, 32)   0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 32, 64, 64)   18496       max_pooling2d[0][0]              
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 32, 64, 64)   256         conv2d_2[0][0]                   
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 32, 64, 64)   36928       batch_normalization_2[0][0]      
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 32, 64, 64)   256         conv2d_3[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 16, 32, 64)   0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 16, 32, 128)  73856       max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 16, 32, 128)  512         conv2d_4[0][0]                   
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 16, 32, 128)  147584      batch_normalization_4[0][0]      
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 16, 32, 128)  512         conv2d_5[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 8, 16, 128)   0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 8, 16, 256)   295168      max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 8, 16, 256)   1024        conv2d_6[0][0]                   
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 8, 16, 256)   590080      batch_normalization_6[0][0]      
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 8, 16, 256)   1024        conv2d_7[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 4, 8, 256)    0           batch_normalization_7[0][0]      
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 4, 8, 256)    590080      max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 4, 8, 256)    1024        conv2d_8[0][0]                   
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 4, 8, 256)    590080      batch_normalization_8[0][0]      
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 4, 8, 256)    1024        conv2d_9[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 2, 4, 256)    0           batch_normalization_9[0][0]      
__________________________________________________________________________________________________
flatten (Flatten)               (None, 2048)         0           max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
dense (Dense)                   (None, 62)           127038      flatten[0][0]                    
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 62)           127038      flatten[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 62)           127038      flatten[0][0]                    
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 62)           127038      flatten[0][0]                    
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 62)           127038      flatten[0][0]                    
==================================================================================================
Total params: 2,993,494
Trainable params: 2,990,550
Non-trainable params: 2,944
__________________________________________________________________________________________________

## Demonstration Video Link :

https://drive.google.com/file/d/1LzY8dp1TeDbbUEoCEc9_tP9T8W8DVXTl/view?usp=sharing
