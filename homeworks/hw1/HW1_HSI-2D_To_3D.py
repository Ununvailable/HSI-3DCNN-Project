# -*- coding: utf-8 -*-
# More generalized script for converting captured HSI datasets
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

dataset_list = {"Blue", "Green", "Red", "Paper"}

for dataset in dataset_list:
    dataset_dir = dataset
    base_dir = r"hsi_datasets/hw1/" + dataset_dir
    list_file = os.listdir(base_dir)
    list_file.sort()

    img_3d = []
    for i,file in enumerate(list_file):
        img = cv2.imread(os.path.join(base_dir,file), 0)
        data = img.mean(axis = 0)
        img_3d.append(img)
    img_3d = np.array(img_3d)
    img_3d = np.transpose(img_3d, (0,2,1)) 

    from scipy.io import savemat
    savemat(base_dir + dataset_dir + ".mat", {
        'DataCube': img_3d
    })