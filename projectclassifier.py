import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

from PIL import Image

import PIL.ImageOps
import os,ssl,time

X = np.load("image.npz")['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses = len(classes)

#splitting the data and scaling it
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 9,train_size = 7500,test_size = 2500)
X_train_scale = X_train/255.0
X_test_scale = X_test/255.0

#fiting the data into the model
clf = LogisticRegression(solver = 'saga', multi_class='multinomial').fit(X_train_scale,y_train)

def get_prediction(image):
    #converting the cv2 image into the pil format
    im_pil = Image.fromarray(image)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)
    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized) 
    pixel_filter = 20 
    min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter) 
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255) 
    max_pixel = np.max(image_bw_resized_inverted) 
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel 
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784) 
    test_pred = clf.predict(test_sample) 
    return test_pred[0]