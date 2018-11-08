import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import load_model
from dataset import get_number_of_diagnosis, get_statistic_of_diagnosis, load_dataset
from generator import generator
from sklearn.model_selection import train_test_split
import pickle as pkl
import os
from generator import generator_png
def count_stat(y_true, y_pred, threshold = 0.9):
    df = pd.DataFrame({str(0): [0, 0 ,0]}, index=['tp', 'fp', 'fn'])

    for j in range(y_pred.shape[1]):
        new_column = [0,0,0]
        for i in range(y_pred.shape[0]):

            if y_pred[i][j] >= threshold:
                if y_true[i][j] == 1:
                    new_column[0] += 1
                else:
                    new_column[1] += 1
            elif y_true[i][j] == 1:
                new_column[2] += 1

        df.insert(loc=df.shape[0]-3, column=str(j+1), value=new_column)

    return df
def count_stat_bin(y_true, y_pred, threshold = 0.9):
    df = pd.DataFrame({str(0): [0, 0 ,0]}, index=['tp', 'fp', 'fn'])


    new_column = [0,0,0]
    for i in range(y_pred.shape[0]):

        if y_pred[i] >= threshold:
            if y_true[i] == 1:
                new_column[0] += 1
            else:
                new_column[1] += 1
        elif y_true[i] == 1:
            new_column[2] += 1

    df.insert(loc=df.shape[0]-3, column=str(1), value=new_column)

    return df

if __name__ == "__main__":
    model = load_model("C:\\Users\\donte_000\\Documents\\GitHub\\ecg_classification\\model_bin.h5")

    pkl_weights = "C:\\Users\\donte_000\\Downloads\\ecg_1078\\weights.pkl"
    pkl_filename = "C:\\Users\\donte_000\\Downloads\\ecg_1078\\dataset_fixed_baseline.pkl"
    raw_dataset = "C:\\Users\\donte_000\\Downloads\\ecg_1078\\data_1078.json"
    num_leads_signal = 1
    win_len = 208
    batch_size = 300

    if os.path.exists(pkl_filename):
        infile = open(pkl_filename, 'rb')
        dataset = pkl.load(infile)
        infile.close()
    else:
        print("Pkl_file is not found.")
        dataset = load_dataset(raw_dataset)

    X, Y = dataset["x"], dataset["y"]

    test_folder_path = "C:\\Users\\donte_000\\Downloads\\ecg_1078\\test_png"
    test_set = next(generator_png(X_png_folder_path=test_folder_path, Y=Y, win_len=win_len, batch_size=batch_size, num_leads_signal=num_leads_signal))

    pred_test = model.predict(test_set[0])

    df = count_stat(test_set[1] ,pred_test, threshold=0.2)
    #df.to_csv("table.csv", sep=';', mode='a', index=False)
    for column in df:
        if df.loc['tp', column] == 0 and df.loc['fp', column] == 0 and df.loc['fn', column] == 0:
            df = df.drop(columns = [column])
        elif (df.loc['tp', column] != 0 and df.loc['fn', column] != 0) or \
                (df.loc['tp', column] != 0 and df.loc['fp', column] == 0 and df.loc['fn', column] == 0):
            print("получилось: " + column)

    print(df)
    #df.to_csv("table3.csv", sep=';', mode='a', index=False)