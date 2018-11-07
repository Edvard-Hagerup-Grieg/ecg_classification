from generator import generator_png
from dataset import load_dataset
import matplotlib.pyplot as plt
from time import time
from PIL import Image
import pickle as pkl
import pandas as pd
import numpy as np
import scipy
import cv2
import os

pkl_filename = "C:\\ecg_new\\dataset_fixed_baseline.pkl"
raw_dataset = "C:\\ecg_new\\data_1078.json"

train_folder_path = "C:\\ecg_new\\train_png"
test_folder_path = "C:\\ecg_new\\test_png"

if os.path.exists(pkl_filename):
    infile = open(pkl_filename, 'rb')
    dataset = pkl.load(infile)
    infile.close()
else:
    print("Pkl_file is not found.")
    dataset = load_dataset(raw_dataset)

X, Y = dataset["x"], dataset["y"]

num_leads_signal = 1
batch_size = 10
win_len = 145

train_generator = generator_png(X_png_folder_path=train_folder_path, Y=Y, win_len=win_len, batch_size=batch_size, num_leads_signal=num_leads_signal)
for i in range(3):
    test_set = next(generator_png(X_png_folder_path=train_folder_path, Y=Y, win_len=145, batch_size=batch_size, num_leads_signal=num_leads_signal))
    for img in test_set[0]:
        cv2.imshow("", img[0, :, :])
        cv2.waitKey(0)