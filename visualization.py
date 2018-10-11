import numpy as np
import pickle as pkl
import os
from generator import generator
from dataset import get_number_of_diagnosis, get_statistic_of_diagnosis, load_dataset


pkl_filename = "C:\\ecg_new\\dataset_fixed_baseline.pkl"
raw_dataset = "C:\\ecg_new\\data_1078.json"
num_leads_signal = 12
batch_size = 100

if os.path.exists(pkl_filename):
    infile = open(pkl_filename, 'rb')
    dataset = pkl.load(infile)
    infile.close()
else:
    print("Pkl_file is not found.")
    dataset = load_dataset(raw_dataset)

X, Y = dataset["x"], dataset["y"]
diagnosis_name = 'left_ventricular_hypertrophy'
diagnosis = get_number_of_diagnosis(diagnosis_name)
get_statistic_of_diagnosis(diagnosis_name, Y)

train_generator = generator(X=X, Y=Y, batch_size=batch_size, diagnosis=diagnosis, num_leads_signal=num_leads_signal)
try:
    for i in range(10):
        test_set = next(generator(X=X, Y=Y, batch_size=batch_size, diagnosis=diagnosis, num_leads_signal=num_leads_signal))
        print("Iteration " + str(i) + ":\n")
        print(np.sum(test_set[1]) / batch_size)
except StopIteration: print("\nError: Too rare class for this batch size.")