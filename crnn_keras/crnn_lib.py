# coding: utf-8

import os 
import sys
sys.path.insert(0, os.getcwd())

from  model import get_model
import keys_keras
from PIL import Image
import difflib
import numpy as np


characters = keys_keras.alphabet_union[:]


# In[3]:

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'models/model_crnn_gru_v1.h5')
# print(os.path.dirname(os.path.dirname(__file__)))
# print(model_path)
# print(os.path.abspath(__file__))



class CRNN_keras(object):
    def __init__(self,model_path=model_path,characters=characters,height=32):
        self.height = height
        self.nClass = len(characters)+1
        self.characters = characters
        self.model,self.basemodel = get_model(self.height,self.nClass)
        self.basemodel.load_weights(model_path)
        
    def predict(self,img):
        im = Image.fromarray(img)
        im = im.convert('L')
        scale = im.size[1] * 1.0 / 32
        w = int(im.size[0] / scale)
        im = im.resize((w, 32))
        img = np.array(im).astype(np.float32) / 255.0
        print(img.shape,scale)
        X = img.reshape((32, w, 1))
        X = np.array([X])
        y_pred = self.basemodel.predict(X)
        y_pred = y_pred[:, 2:, :]
        out = self.decode(y_pred)  ##
        
        if len(out) > 0:
            while out[0] == u'。':
                if len(out) > 1:
                    out = out[1:]
                else:
                    break
        return out
    
    def decode(self,pred):
        charactersS = self.characters + u' '
        t = pred.argmax(axis=2)[0]
        length = len(t)
        char_list = []
        n = self.nClass-1
        for i in range(length):
            if t[i] != n and (not (i > 0 and t[i - 1] == t[i])):
                char_list.append(charactersS[t[i]])
        return u''.join(char_list)
    
    def load_weights(self,model_path):
        self.basemodel.load_weights(model_path)


if __name__ == '__main__':
    import cv2
    crnn = CRNN_keras()
    img_path = './test/"依那普利氢氯噻嗪.png'
    image = cv2.imread(img_path)
    text = crnn.recognize(image)
    print(text)




