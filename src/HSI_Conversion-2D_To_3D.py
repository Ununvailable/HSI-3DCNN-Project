# -*- coding: utf-8 -*-
"""
Created on Thu May  1 12:57:37 2025

@author: stust
"""
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

dataset_dir = r"Spectrum-Simplified"
base_dir = r"hsi_datasets/v303/" + dataset_dir
list_file = os.listdir(base_dir)
list_file.sort()

img_3d = []
for i,file in enumerate(list_file):
    img = cv2.imread(os.path.join(base_dir,file), 0)
    data = img.mean(axis = 0)
    img_3d.append(img)
img_3d = np.array(img_3d)              # 目前格式為 (Y, B, X)
# MATLAB 與 NumPy 的陣列記憶體順序（row-major vs column-major）
# 不同，# 因此有時會出現資料顯示順序怪異的情況。若希望在 MATLAB 
# 中正確顯示影像，須事先轉成 (Y, X, B)
img_3d = np.transpose(img_3d, (0,2,1)) 

# MAT檔案中會儲存一個變數名稱為 "DataCube"，其對應的值是 img_3d。
# 當載入這個 .mat 檔案後，會看到一個變數叫 DataCube，資料就是原本的 img_3d。
from scipy.io import savemat
savemat(r"E:/Liam/HSI-3DCNN-Project/hsi_datasets/v303/" + dataset_dir + ".mat", {
    'DataCube': img_3d
})