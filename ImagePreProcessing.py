# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 08:03:14 2019

@author: Ivana Situmorang
"""

import cv2
import os
import numpy as np
from tqdm import tqdm


image_height = 100
image_width = 100
img_type = 0

def set_image_size(height = 100, width = 100):
    global image_height
    global image_width
    image_height = height
    image_width = width
    return

def get_image_size():
    return (image_width, image_height)

def set_image_type(image_type = img_type):
    global img_type
    img_type = image_type
    return

def get_image_type():
    return img_type

def load_image(path_image, image_type=0, image_width=100, image_height=100):
    if image_type == 0:
        tmp_img = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
    
    else :
        tmp_img = cv2.imread(path_image, cv2.IMREAD_COLOR)
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
    
    tmp_img = cv2.resize(tmp_img, (image_width, image_height)) 
    return tmp_img

def resize(image, height, width):
    return cv2.resize(image, (width, height))

def flip_horizontal(image) :
    return cv2.flip(image, 0)

def flip_vertical(image) :
    return cv2.flip(image, 1)

def rotate(image, angle):
    rows = image.shape[0]
    cols = image.shape[1]
    image_center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(image_center, angle, 1)

    rotated_image = cv2.warpAffine(image, M, (cols, rows), borderValue=(255,255,255))
    
    return rotated_image

def scale(image, crop_size):
    tmp_img = image[0:image.shape[1]- crop_size[1], 0:image.shape[0]- crop_size[0]]
    tmp_img = cv2.resize(tmp_img, (image.shape[1], image.shape[0]))
    return tmp_img

def translate(image, translation_distance):
    rows = image.shape[0]
    cols = image.shape[1]
    M = np.float32([[1, 0, translation_distance[0]], [0, 1, translation_distance[1]]])
    tmp_img = cv2.warpAffine(image, M, (cols, rows), borderValue=(255,255,255))
    return tmp_img

def save_image(path_image, image):
    if get_image_type() == 0:
        cv2.imwrite(path_image, image)
    if get_image_type() == 1:
        cv2.imwrite(path_image, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(path_image)
    return

def augment_flip(dataset_path, target_path):
    if os.path.exists(target_path) == False:
            os.makedirs(target_path)
    for folder in tqdm(os.listdir(dataset_path)):
        path_folder = os.path.join(dataset_path, folder)
        new_path_folder = os.path.join(target_path, folder)
        if os.path.exists(new_path_folder) == False:
            os.makedirs(new_path_folder)
        counter=0
        for image_file in tqdm(os.listdir(path_folder)):
            path_file = os.path.join(path_folder, image_file)
            image = load_image(path_file, get_image_type())
            save_image(os.path.join(new_path_folder, "fliph-"+str(counter))+"."+image_file.split(".")[-1], flip_horizontal(image))
            counter+=1
            save_image(os.path.join(new_path_folder, "flipv-"+str(counter))+"."+image_file.split(".")[-1], flip_vertical(image))
            counter+=1
    return
    
def augment_rotate(dataset_path, target_path):
    if os.path.exists(target_path) == False:
            os.makedirs(target_path)
    for folder in tqdm(os.listdir(dataset_path)):
        path_folder = os.path.join(dataset_path, folder)
        new_path_folder = os.path.join(target_path, folder)
        if os.path.exists(new_path_folder) == False:
            os.makedirs(new_path_folder)
        counter=0
        for image_file in tqdm(os.listdir(path_folder)):
            path_file = os.path.join(path_folder, image_file)
            image = load_image(path_file, get_image_type())
            save_image(os.path.join(new_path_folder, "rotate45-"+str(counter))+"."+image_file.split(".")[-1], rotate(image, 45))
            counter+=1
            save_image(os.path.join(new_path_folder, "rotate90-"+str(counter))+"."+image_file.split(".")[-1], rotate(image, 90))
            counter+=1
    return

def augment_scale(dataset_path, target_path):
    if os.path.exists(target_path) == False:
            os.makedirs(target_path)
    for folder in tqdm(os.listdir(dataset_path)):
        path_folder = os.path.join(dataset_path, folder)
        new_path_folder = os.path.join(target_path, folder)
        if os.path.exists(new_path_folder) == False:
            os.makedirs(new_path_folder)
        counter=0
        for image_file in tqdm(os.listdir(path_folder)):
            path_file = os.path.join(path_folder, image_file)
            image = load_image(path_file, get_image_type())
            adder=0
            height = image.shape[0]
            width = image.shape[0]
            if (height/10 >= 100):
               adder=500 
            elif (height/10 >= 10):
               adder=50
            else :
                adder=25
            
            for i in range(0, int(height/2)+1, adder):
                for j in range(0, int(width/2)+1, adder):
                    if (i!=0) or (j!=0):
                        save_image(os.path.join(new_path_folder, "scale-"+str(counter))+"-("+str(i)+","+str(j)+")."+image_file.split(".")[-1], scale(image, (i,j)))
                        counter+=1
    return
    
def augment_translate(dataset_path, target_path):
    if os.path.exists(target_path) == False:
            os.makedirs(target_path)
    for folder in tqdm(os.listdir(dataset_path)):
        path_folder = os.path.join(dataset_path, folder)
        new_path_folder = os.path.join(target_path, folder)
        if os.path.exists(new_path_folder) == False:
            os.makedirs(new_path_folder)
        counter=0
        for image_file in tqdm(os.listdir(path_folder)):
            path_file = os.path.join(path_folder, image_file)
            image = load_image(path_file, get_image_type())
            adder=0
            height = image.shape[0]
            width = image.shape[0]
            if (height/10 >= 100):
               adder=250 
            elif (height/10 >= 10):
               adder=25
            else :
                adder=15
            
            for i in range(0, int(height/4)+1, adder):
                for j in range(0, int(width/4)+1, adder):
                    if (i!=0) and (j!=0):
                        save_image(os.path.join(new_path_folder, "translate-"+str(counter))+"-("+str(i*-1)+","+str(j*-1)+")."+image_file.split(".")[-1], translate(image, (i*-1,j*-1)))
                        counter+=1
                    if (i!=0) or (j!=0):
                        save_image(os.path.join(new_path_folder, "translate-"+str(counter))+"-("+str(i)+","+str(j)+")."+image_file.split(".")[-1], translate(image, (i,j)))
                        counter+=1
                    if (i!=0):
                        save_image(os.path.join(new_path_folder, "translate-"+str(counter))+"-("+str(i*-1)+","+str(j)+")."+image_file.split(".")[-1], translate(image, (i*-1,j)))
                        counter+=1
                    if (j!=0):
                        save_image(os.path.join(new_path_folder, "translate-"+str(counter))+"-("+str(i)+","+str(j*-1)+")."+image_file.split(".")[-1], translate(image, (i,j*-1)))
                        counter+=1
                        
    return


def label_data(dataset_path, target_path):
    if os.path.exists(target_path) == False:
            os.makedirs(target_path)
    for folder in tqdm(os.listdir(dataset_path)):
        path_folder = os.path.join(dataset_path, folder)
        new_path_folder = os.path.join(target_path, folder)
        if os.path.exists(new_path_folder) == False:
            os.makedirs(new_path_folder)
        index = 1
        for file in tqdm(os.listdir(path_folder)):
            path_file = os.path.join(path_folder, file)
            img = load_image(path_file, get_image_type())
            new_name = folder+"."+str(index)+"."+file.split(".")[-1]
            save_image(os.path.join(new_path_folder, new_name), img)
            index+=1     
    return

def augment_data(dataset_path, target_path):
    augment_flip(dataset_path, target_path)
    augment_rotate(dataset_path, target_path)
    augment_scale(dataset_path, target_path)
    augment_translate(dataset_path, target_path)
    return

def segment_canny(arr_image, height=100, width=100):
    temp=arr_image.copy()
    if arr_image.shape != (height,width):
        temp = arr_image.reshape(height,width)
    if arr_image.dtype != np.uint8:
        temp=temp*255
        temp=temp.astype(np.uint8)
    
    temp=cv2.Canny(temp, 50, 100)
    temp=temp/255
    return temp

def segment_region_based(arr_image, height=100, width=100):
    temp = arr_image.copy()
    temp = temp.reshape(arr_image.shape[0]*arr_image.shape[1])
    for i in range(temp.shape[0]):
        if temp[i] > temp.mean():
            temp[i] = 1
        else:
            temp[i] = 0 
    temp = temp.reshape(height,width)
    return temp
    