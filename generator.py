import pickle as pkl
import numpy as np
import os


pkl_filename = "C:\\ecg_new\\weights.pkl"

#со значимостью пациентов
def generator(X,Y, win_len, batch_size, num_leads_signal=12):
    """
    :param X: экг (12 отведений)
    :param Y: соответствующие им диагнозвы
    :param win_len: длина интервала экг
    :param batch_size: сколько пациентов брать в батч
    :param num_leads_signal: сколько брать отведений
    """

    weights = []
    if os.path.exists(pkl_filename):
        infile = open(pkl_filename, 'rb')
        weights = pkl.load(infile)
        infile.close()
    else:
        for i in range(Y.shape[0]): #для каждого пациента
            print("\nПациент: ", i)
            weight = 0
            for j in range(Y.shape[1]):
                u_j = (sum(Y[:, j]) / Y.shape[0]) #частота встречаемости единицы
                weight += Y[i,j]*(1. - u_j) + (1 - Y[i,j])*u_j #вес пациента

            print("Вес пациента: ", weight)
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

#балансировка по одному диагнозу (Y для бинарной классификации по этому диагнозу)
def generator_bin(X,Y, batch_size, diagnosis=0, num_leads_signal=12):
    """
    :param X: экг (12 отведений)
    :param Y: соответствующие им диагнозвы
    :param batch_size: сколько пациентов брать в батч
    :param diagnosis: по какому диагнозу сбалансировать батч
    :param
    """

    #X = (X_health, X_ill)
    X = (X[Y[:, diagnosis] == 0, :, :], X[Y[:, diagnosis] == 1, :, :])
    if min(X[0].shape[0], X[1].shape[0]) < (0.5 * batch_size): return

    while True:
        batch_x = []
        batch_y = []
        for i in range(0, batch_size):
            patient_status = np.random.choice([0,1])
            rand_pacient_id = np.random.randint(0, X[patient_status].shape[0])

            x = X[patient_status][rand_pacient_id, :, 0:num_leads_signal]
            y = patient_status
            batch_x.append(x)
            batch_y.append(y)

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        yield (batch_x, batch_y)

def generator_cut(X,Y, win_len, batch_size, diagnosis=0, num_leads_signal=12):
    """
    :param X: экг (12 отведений)
    :param Y: соответствующие им диагнозвы
    :param win_len: длина интервала экг
    :param batch_size: сколько пациентов брать в батч
    :param diagnosis: по какому диагнозу сбалансировать батч
    :param
    """

    all_ecg_len = X.shape[1]

    #X = (X_health, X_ill)
    X = (X[Y[:, diagnosis] == 0, :, :], X[Y[:, diagnosis] == 1, :, :])

    while True:
        batch_x = []
        batch_y = []
        for i in range(0, batch_size):
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