import numpy as np


def generator(X,Y, batch_size, test_split=0.2, diagnosis=0, num_leads_signal=12):
    """
    :param X: экг (12 отведений)
    :param Y: соответствующие им диагнозвы
    :param batch_size: сколько пациентов брать в батч
    :param test_split: доля тестовых данных в датасете
    :param diagnosis: по какому диагнозу сбалансировать батч
    :param
    """

    #X = (X_health, X_ill)
    X = (X[Y[:, diagnosis] == 0, :, :], X[Y[:, diagnosis] == 1, :, :])
    if min(X[0].shape[0], X[1].shape[0]) < (0.5 * batch_size): return

    # test_x = []
    # test_y = []

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

def generator_cut(X,Y, win_len, batch_size, test_split=0.2, diagnosis=0, num_leads_signal=12):
    """
    :param X: экг (12 отведений)
    :param Y: соответствующие им диагнозвы
    :param win_len: длина интервала экг
    :param batch_size: сколько пациентов брать в батч
    :param test_split: доля тестовых данных в датасете
    :param diagnosis: по какому диагнозу сбалансировать батч
    :param
    """

    all_ecg_len = X.shape[1]

    #X = (X_health, X_ill)
    X = (X[Y[:, diagnosis] == 0, :, :], X[Y[:, diagnosis] == 1, :, :])

    # test_x = []
    # test_y = []

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