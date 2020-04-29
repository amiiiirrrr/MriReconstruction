import pickle
import tensorlayer as tl
import numpy as np
import os
import nibabel as nib
import glob
import zipfile
import cv2
from PIL import Image
import os

training_data_path = "data\MICCAI13_SegChallenge\Training_100"
testing_data_path = "data\MICCAI13_SegChallenge\Testing_50"
validation_data_path = "data\MICCAI13_SegChallenge\Validation"
val_ratio = 0.2
seed = 100
preserving_ratio = 0.1 # filter out 2d images containing < 10% non-zeros

X_train = []
for f in os.listdir(training_data_path):
    img_path = os.path.join(training_data_path,f)
    img = Image.open(img_path)
    img_3d_max = np.max(img)
    img = img / img_3d_max * 255
    img = img / 127.5 - 1
    img_2d = np.transpose(img, (1, 0))
    a = img_2d.shape[0]
    b = img_2d.shape[1]
    if a == 256 & b == 256 :
        X_train.append(img_2d)


X_test = []
for f in os.listdir(testing_data_path):
    img_path = os.path.join(testing_data_path,f)
    img = Image.open(img_path)
    img_3d_max = np.max(img)
    img = img / img_3d_max * 255
    img = img / 127.5 - 1
    img_2d = np.transpose(img, (1, 0))
    a = img_2d.shape[0]
    b = img_2d.shape[1]
    if a == 256 & b == 256 :
        X_test.append(img_2d)
    
    
X_val = []
for f in os.listdir(validation_data_path):
    img_path = os.path.join(validation_data_path,f)
    img = Image.open(img_path)
    img_3d_max = np.max(img)
    img = img / img_3d_max * 255
    img = img / 127.5 - 1
    img_2d = np.transpose(img, (1, 0))
    a = img_2d.shape[0]
    b = img_2d.shape[1]
    if a == 256 & b == 256 :
        X_val.append(img_2d)
    

#imshow(X_train[45,:,:,0])
X_train = np.asarray(X_train)
X_train = X_train[:, :, :, np.newaxis]
X_val = np.asarray(X_val)
X_val = X_val[:, :, :, np.newaxis]
X_test = np.asarray(X_test)
X_test = X_test[:, :, :, np.newaxis]
#X_train = X_train.astype(np.float32)
#X_val = X_val.astype(np.float32)
#X_test = X_test.astype(np.float32)
# save data into pickle format
data_saving_path = 'data/MICCAI13_SegChallenge/'
tl.files.exists_or_mkdir(data_saving_path)

print("save training set into pickle format")
with open(os.path.join(data_saving_path, 'training.pickle'), 'wb') as f:
    pickle.dump(X_train, f, protocol=4)

print("save validation set into pickle format")
with open(os.path.join(data_saving_path, 'validation.pickle'), 'wb') as f:
    pickle.dump(X_val, f, protocol=4)

print("save test set into pickle format")
with open(os.path.join(data_saving_path, 'testing.pickle'), 'wb') as f:
    pickle.dump(X_test, f, protocol=4)

print("processing data finished!")


