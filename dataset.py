import BaselineWanderRemoval as bwr
import pickle as pkl
import numpy as np
import json
import os


leads_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
FREQUENCY_OF_DATASET = 250
pkl_filename = "C:\\ecg_new\\dataset_fixed_baseline.pkl"
pkl_dictionary= "C:\\ecg_new\\dictionary_of_diagnoses.pkl"
raw_dataset_path = "C:\\ecg_new\\data_1078.json"

def load_raw_dataset(raw_dataset):

    with open(raw_dataset, 'r') as f:
        data = json.load(f)
    if not os.path.exists(pkl_dictionary):
        seve_dictionary_of_diagnoses_to_pkl(data)
    infile = open(pkl_dictionary, 'rb')
    dictionary = pkl.load(infile)
    infile.close()

    X=[]
    Y=[]
    for case_id in data.keys():
        leads = data[case_id]['Leads']
        if len(data[case_id]['Leads'].keys()) != 12: continue

        diagnosis = data[case_id]['StructuredDiagnosisDoc']
        if len(diagnosis.keys()) != 212: continue

        x = []
        signal_len = 2500
        for i in range(len(leads_names)):
            lead_name = leads_names[i]
            if len(leads[lead_name]['Signal']) > signal_len:
                x.append(leads[lead_name]['Signal'][::2])
            else:
                x.append(leads[lead_name]['Signal'])

        y = []
        for i in dictionary.keys():
            y.append(diagnosis[i])
        y = np.where(y, 1, 0)

        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
    X = np.swapaxes(X, 1, 2)

    # X.shape = (1073, 2500, 12)
    # Y.shape = (1073, 212)

    return {"x": X, "y": Y}

def fix_baseline_and_save_to_pkl(xy):
    print("start fixing baseline in the whole dataset. It may take some time, wait...")
    X = xy["x"]
    for i in range(X.shape[0]):
        print(str(i))

        for j in range(X.shape[2]):
            X[i, :, j] = bwr.fix_baseline_wander(X[i, :, j], FREQUENCY_OF_DATASET)
    xy['x']=X
    outfile = open(pkl_filename, 'wb')
    pkl.dump(xy, outfile)
    outfile.close()
    print("dataset saved, number of pacients = " + str(len(xy['x'])))

def load_dataset(raw_dataset=raw_dataset_path, fixed_baseline=True):
    if fixed_baseline is True:
        print("you selected FIXED BASELINE WANDERING")
        if os.path.exists(pkl_filename): # если файл с предобработанным датасетом уже есть, не выполняем предобработку
            infile = open(pkl_filename, 'rb')
            dataset_with_fixed_baseline = pkl.load(infile)
            infile.close()
            return dataset_with_fixed_baseline
        else:
            xy = load_raw_dataset(raw_dataset) # если файл с обработанным датасетом еще не создан, создаем
            fix_baseline_and_save_to_pkl(xy)
            infile = open(pkl_filename, 'rb')
            dataset_with_fixed_baseline = pkl.load(infile)
            infile.close()
            return dataset_with_fixed_baseline
    else:
        print("you selected NOT fixied BASELINE WANDERING")
        return load_raw_dataset(raw_dataset)

def seve_dictionary_of_diagnoses_to_pkl(data):
    diagnoses = data['60909568']['StructuredDiagnosisDoc'].keys()
    numbers = list(range(212))
    dictionary = dict(zip(diagnoses, numbers))
    outfile = open(pkl_dictionary, 'wb')
    pkl.dump(dictionary, outfile)
    outfile.close()

def get_number_of_diagnosis(diagnosis):
    infile = open(pkl_dictionary, 'rb')
    dictionary = pkl.load(infile)
    try: number = dictionary[diagnosis]
    except KeyError:
        print("This diagnosis is not correct.")
        number = None
    return number

def get_statistic_of_diagnosis(diagnosis, Y):
    num_of_patient = Y.shape[0]
    number_of_diagnosis = get_number_of_diagnosis(diagnosis)
    print("\nNumber of diagnosis: " + str(number_of_diagnosis))
    print("\nNumber of sick patients: " + str(sum(Y[:,number_of_diagnosis])) + " / " + str(num_of_patient))
    print("\nSick patient frequency in the data: " + str(sum(Y[:,number_of_diagnosis]) / num_of_patient))

if __name__ == "__main__":
    xy = load_dataset()
    print(xy)