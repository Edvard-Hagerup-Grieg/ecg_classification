import pandas as pd
import pickle as pkl
import os
from generator import generator
from dataset import get_number_of_diagnosis, get_statistic_of_diagnosis, load_dataset


pkl_weights = "C:\\ecg_new\\weights.pkl"
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

infile = open(pkl_weights, 'rb')
weights = pkl.load(infile)
infile.close()

headers = ['frequency']
for w, n in weights:
    headers.append(w)
print(headers)

columns = []
column = []
for diagnosis in range(Y.shape[1]):
    column.append(sum(Y[:, diagnosis]) / Y.shape[0])
columns.append(column)

train_generator = generator(X=X, Y=Y, win_len=200, batch_size=batch_size, num_leads_signal=num_leads_signal)
for i in range(1):
    test_set = next(generator(X=X, Y=Y, win_len=200, batch_size=batch_size, num_leads_signal=num_leads_signal))
    for y in test_set[1]:
        for p in y:
            columns.append(list(p))


diction = dict(zip(headers, columns))
table = pd.DataFrame(diction)

path = "C:\\ecg_new\\table.csv"
table.to_csv(path, sep=';', mode='a', index=False)