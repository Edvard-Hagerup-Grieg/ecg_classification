import pickle as pkl
import numpy as np
import cv2
import os


leads_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
pkl_filename = "C:\\ecg_new\\weights.pkl"

START = 185

def generator(X, Y, win_len, batch_size, num_leads_signal=12, pkl_filename = pkl_filename):
    """
    :param X: экг (12 отведений)
    :param Y: соответствующие им диагнозвы
    :param win_len: длина интервала экг
    :param batch_size: сколько пациентов брать в батч
    :param num_leads_signal: сколько брать отведений
    :param pkl_filename: куда записать веса пациентов (должен быть разный путь для test и train)
    """

    weights = []
    if os.path.exists(pkl_filename):
        infile = open(pkl_filename, 'rb')
        weights = pkl.load(infile)
        infile.close()
    else:
        for i in range(Y.shape[0]):
            weight = 0
            for j in range(Y.shape[1]):
                u_j = (sum(Y[:, j]) / Y.shape[0]) #частота встречаемости единицы
                weight += Y[i,j]*(1. - u_j) + (1 - Y[i,j])*u_j #вес пациента

            weights.append([weight, i])
            weights = sorted(weights, key=lambda tup: tup[0])

        outfile = open(pkl_filename, 'wb')
        pkl.dump(weights, outfile)
        outfile.close()

    all_ecg_len = X.shape[1]

    while True:
        batch_x = []
        batch_y = []
        for i in range(0, batch_size):
            starting_position = np.random.randint(0, all_ecg_len - win_len)
            ending_position = starting_position + win_len

            patient = 0
            MAX_WEIGHT = weights[len(weights) - 1][0]
            MIN_WEIGHT = weights[0][0]
            choice = np.random.uniform(MIN_WEIGHT,MAX_WEIGHT)
            for j in range(len(weights) - 1):
                if choice >= weights[j][0] and choice < weights[j + 1][0]:
                    patient = weights[j][1]

            x = X[patient, starting_position:ending_position, 0:num_leads_signal]
            y = Y[patient, :]
            batch_x.append(x)
            batch_y.append(y)

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        yield (batch_x, batch_y)

def generator_bin(X,Y, batch_size, diagnosis=0, num_leads_signal=12, win_len=200, cut=False):
    """
    :param X: экг (12 отведений)
    :param Y: соответствующие им диагнозвы
    :param batch_size: сколько пациентов брать в батч
    :param diagnosis: по какому диагнозу сбалансировать батч
    :param num_leads_signal: сколько отведений брать
    :param win_len: длина интервала экг
    :param cut: True - брать все экг, False - брать часть экг заданной длины
    """

    all_ecg_len = X.shape[1]

    #X = (X_health, X_ill)
    X = (X[Y[:, diagnosis] == 0, :, :], X[Y[:, diagnosis] == 1, :, :])

    while True:
        batch_x = []
        batch_y = []
        for i in range(0, batch_size):
            starting_position = 0
            ending_position = all_ecg_len
            if cut is True:
                starting_position = np.random.randint(0, all_ecg_len - win_len)
                ending_position = starting_position + win_len

            patient_status = np.random.choice([0,1])
            rand_pacient_id = np.random.randint(0, X[patient_status].shape[0])

            x = X[patient_status][rand_pacient_id, starting_position:ending_position, 0:num_leads_signal]
            y = patient_status
            batch_x.append(x)
            batch_y.append(y)

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        yield (batch_x, batch_y)

def generator_png(X_png_folder_path,Y, batch_size, num_leads_signal=12, win_len=145):
    """
    :param X_png_folder_path: путь к папке с картинками (train или test)
    :param Y: соответствующие им диагнозвы
    :param batch_size: сколько пациентов брать в батч
    :param num_leads_signal: сколько отведений брать
    :param win_len: длина картинки в пикселях
    """
    infile = open(X_png_folder_path + "\\patient_num.pkl", 'rb')
    num_patient = pkl.load(infile)
    infile.close()

    while True:
        batch_x = []
        batch_y = []
        for i in range(0, batch_size):
            patient = np.random.randint(0, len(num_patient))

            x = []
            for lead in range(num_leads_signal):
                name = X_png_folder_path + "\\" + str(num_patient[patient]) + "_" + leads_names[lead] + ".png"
                img = cv2.imread(name, 0)

                crop_img = img[:145, START:START + win_len]
                arr = np.array(crop_img)

                x.append(arr)

            y = Y[num_patient[patient], :]

            batch_x.append(x)
            batch_y.append(y)

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        batch_x = np.swapaxes(batch_x, 1, 3)

        yield (batch_x, batch_y)