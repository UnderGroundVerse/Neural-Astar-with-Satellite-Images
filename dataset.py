import pandas as pd
import numpy as np
import cv2 as cv
from PIL import Image
import os

def load_metadata(csv_file_path):
    return pd.read_csv(f'{csv_file_path}')

def load_images_paths(csv_file):
    return (csv_file['sat_image_path'] , csv_file['mask_image_path'])

def prepare_trainingdata(x_train_paths,y_train_paths, img_size=(256,256),number_of_buffer=2000):
    
    x_train = []
    y_train = []
    
    for path in x_train_paths[:number_of_buffer]:
        img = Image.open(path)
        img = img.resize(img_size)
        x_train.append(np.asarray(img))
    for path in y_train_paths[:number_of_buffer]:
        img = Image.open(path)
        img = img.resize(img_size)
        img = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)
        y_train.append(img)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    x_train = x_train / 255.0
    y_train = y_train / 255.0
    
    return (x_train, y_train)
    
def prepare_testdata(folder_path, img_size=(256,256)):
    test_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path)
            img = img.resize(img_size)
            test_images.append(np.asarray(img))

    test_images = np.array(test_images)
    test_images = test_images / 255.0
    return test_images
    
def porcess_masked_for_training(csv_file, img_size=(256,256),buffer_size=2000):
    mask_paths = csv_file['mask_path']
    mask_images = []

    for path in mask_paths[:buffer_size]:
        img = Image.open(path)
        img = img.resize(img_size)
        img = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)
        mask_images.append(img)

    mask_images = np.array(mask_images)
    mask_images = mask_images / 255.0
    threshold = 0.1
    mask_images = np.where(mask_images > threshold, 1, 0)
    return mask_images


    
        
