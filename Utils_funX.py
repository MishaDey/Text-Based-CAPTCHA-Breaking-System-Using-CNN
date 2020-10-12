import numpy as np
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from tqdm import tqdm # tqdm means progress ->> used to draw the progress bar
from Captcha_Sequence_Generator import Captcha_Sequence_Generator

def Decode_Y_Val(Y_val,characters):
    char_classes=np.argmax(Y_val,axis=2)[:,0]
    String=''.join(characters[ind] for ind in char_classes)
    return String,char_classes


def evaluate_metrics(model,characters,n_batch=100):
    # Overall Accuracy = Net batch Acc / no of batches
    net_batch_acc = 0.0
    with tqdm(Captcha_Sequence_Generator(characters, batch_size=128, steps=100)) as progress_bar: # Total test size = 128 * 100
        for X, y in progress_bar:
            y_pred = model.predict(X)
            y_pred = np.argmax(y_pred, axis=-1).T
            y_true = np.argmax(y, axis=-1).T

            net_batch_acc += (y_true == y_pred).all(axis=-1).mean()
    return net_batch_acc / n_batch
