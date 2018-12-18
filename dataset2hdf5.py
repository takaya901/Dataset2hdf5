#%%
import numpy as np
import glob
import h5py
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

dataset_path = '/Users/takaya/Documents/Research/Dataset'

#%%
orgs = []
masks = []
test_orgs = []
test_masks = []

print('without target img')
files = glob.glob(dataset_path+'/train/without/*.jpg')
img_num = len(files)
for i in range(img_num):
    print(i)
    img = load_img(dataset_path+'/train/without/' + str(i) + '.jpg', target_size=(64,64))
    imgarray = img_to_array(img)
    orgs.append(imgarray)
    
print('with target img')
files = glob.glob(dataset_path+'/train/with/*.jpg')
img_num = len(files)
for i in range(img_num):
    print(i)
    img = load_img(dataset_path+'/train/with/' + str(i) + '.jpg', target_size=(64,64))
    imgarray = img_to_array(img)
    masks.append(imgarray)

print('without target img')
files = glob.glob(dataset_path+'/test/without/*.jpg')
img_num = len(files)
for i in range(img_num):
    print(i)
    img = load_img(dataset_path+'/test/without/' + str(i) + '.jpg', target_size=(64,64))
    imgarray = img_to_array(img)
    test_orgs.append(imgarray)

print('with target img')
files = glob.glob(dataset_path+'/test/with/*.jpg')
img_num = len(files)
for i in range(img_num):
    print(i)
    img = load_img(dataset_path+'/test/with/' + str(i) + '.jpg', target_size=(64,64))
    imgarray = img_to_array(img)
    test_masks.append(imgarray)

#%%
imgs = np.array(orgs)
gimgs = np.array(masks)
vimgs = np.array(test_orgs)
vgimgs= np.array(test_masks)
print('shapes')
print('org imgs  : ', imgs.shape)
print('mask imgs : ', gimgs.shape)
print('test org  : ', vimgs.shape)
print('test tset : ', vgimgs.shape)

outh5 = h5py.File(dataset_path + '/dataset.hdf5', 'w')
outh5.create_dataset('TrainWithoutTarget', data=imgs)
outh5.create_dataset('TrainWithTarget', data=gimgs)
outh5.create_dataset('TestWithoutTarget', data=vimgs)
outh5.create_dataset('TestWithTarget', data=vgimgs)
outh5.flush()
outh5.close()