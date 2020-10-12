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

## Model.Summary() 

    Total params: 2,993,494
    Trainable params: 2,990,550
    Non-trainable params: 2,944

#### Overall Accuracy = 0.95578125

## Demonstration Video Link :

https://drive.google.com/file/d/1LzY8dp1TeDbbUEoCEc9_tP9T8W8DVXTl/view?usp=sharing
