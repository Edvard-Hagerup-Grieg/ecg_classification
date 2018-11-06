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


pkl_weights = "C:\\ecg_new\\weights.pkl"
pkl_filename = "C:\\ecg_new\\dataset_fixed_baseline.pkl"
raw_dataset = "C:\\ecg_new\\data_1078.json"

png_dataset1 = "C:\\ecg_new\\png1.png"
png_dataset2 = "C:\\ecg_new\\png2.png"
num_leads_signal = 12

if os.path.exists(pkl_filename):
    infile = open(pkl_filename, 'rb')
    dataset = pkl.load(infile)
    infile.close()
else:
    print("Pkl_file is not found.")
    dataset = load_dataset(raw_dataset)

X, Y = dataset["x"], dataset["y"]

patient = 200
png_dataset = png_dataset2
height = 128
width = 2048
dpi = 100

START = 185

fig, ax = plt.subplots(nrows=1, ncols=1, dpi=dpi, figsize=(width / dpi, height / dpi))
ax.plot(X[patient, :, 0])
plt.axis('off')
fig.savefig(png_dataset, bbox_inches="tight", dpi=dpi)
plt.close(fig)
plt.show()