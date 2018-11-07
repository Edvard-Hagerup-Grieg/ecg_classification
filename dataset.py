import BaselineWanderRemoval as bwr
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import shutil
import json
import os


leads_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
FREQUENCY_OF_DATASET = 250

png_dataset_folder = "C:\\ecg_new\\dataset_png"
pkl_filename = "C:\\ecg_new\\dataset_fixed_baseline.pkl"
pkl_dictionary= "C:\\ecg_new\\dictionary_of_diagnoses.pkl"
raw_dataset_path = "C:\\ecg_new\\data_1078.json"

train_folder_path = "C:\\ecg_new\\train_png"
test_folder_path = "C:\\ecg_new\\test_png"


def load_raw_dataset(raw_dataset, save_png=False):

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

    if save_png is True:
        save_dataset_in_png(X)

    return {"x": X, "y": Y}

def fix_baseline_and_save_to_pkl(xy):
    print("Start fixing baseline in the whole dataset. It may take some time, wait...")
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

def save_dataset_in_png(X):
    if not os.listdir(png_dataset_folder):
        print("Start saving dataset in images. It may take some time, wait...")

        height = 128
        width = 2048
        dpi = 100

        for patient in range(X.shape[0]):
            print(patient)
            for lead in range(X.shape[2]):
                name = png_dataset_folder + "\\" + str(patient) + "_" + leads_names[lead] + ".png"

                fig, ax = plt.subplots(nrows=1, ncols=1, dpi=dpi, figsize=(width / dpi, height / dpi))
                ax.plot(X[patient, :, lead])
                plt.axis('off')
                fig.savefig(name, bbox_inches="tight", dpi=dpi)
                plt.close(fig)
                plt.show()
    else:
        print("Folder is not empty.")

def load_dataset(raw_dataset=raw_dataset_path, fixed_baseline=True, save_png=False):
    if fixed_baseline is True:
        print("you selected FIXED BASELINE WANDERING")
        if os.path.exists(pkl_filename): # если файл с предобработанным датасетом уже есть, не выполняем предобработку
            infile = open(pkl_filename, 'rb')
            dataset_with_fixed_baseline = pkl.load(infile)
            infile.close()

            if save_png is True:
                save_dataset_in_png(dataset_with_fixed_baseline["x"])

            return dataset_with_fixed_baseline
        else:
            xy = load_raw_dataset(raw_dataset) # если файл с обработанным датасетом еще не создан, создаем
            fix_baseline_and_save_to_pkl(xy)
            infile = open(pkl_filename, 'rb')
            dataset_with_fixed_baseline = pkl.load(infile)
            infile.close()

            if save_png is True:
                save_dataset_in_png(dataset_with_fixed_baseline["x"])

            return dataset_with_fixed_baseline
    else:
        print("you selected NOT fixied BASELINE WANDERING")
        return load_raw_dataset(raw_dataset, save_png=save_png)

def train_test_split_png(test_size=0.33, data_folder=png_dataset_folder, train_folder_path=train_folder_path, test_folder_path=test_folder_path):
    num_patient = 1073
    test_num = int(num_patient * test_size)

    all_list_num = list(range(num_patient))
    test_list_num = np.random.randint(0, num_patient-1, test_num)
    for patient in range(num_patient):
        if patient in test_list_num: folder = test_folder_path
        else: folder = train_folder_path

        for lead in leads_names:
            name1 = data_folder + "\\" + str(patient) + "_" + lead + ".png"
            name2 = folder + "\\" + str(patient) + "_" + lead + ".png"
            shutil.copy(name1, name2)

    outfile = open(test_folder_path + "\\patient_num.pkl", 'wb')
    pkl.dump(test_list_num, outfile)
    outfile.close()

    outfile = open(train_folder_path + "\\patient_num.pkl", 'wb')
    pkl.dump([item for item in all_list_num if item not in test_list_num], outfile)
    outfile.close()


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

    if sum(Y[:,number_of_diagnosis]) == 0: return 1
    else: return 0

def plot_hist(Y, frequency=0.):
    infile = open(pkl_dictionary, 'rb')
    dictionary = pkl.load(infile)

    hist = []
    for diagnosis in dictionary.keys():
        value = sum(Y[:, dictionary[diagnosis]]) / Y.shape[0]
        if value > frequency:
            hist.append([sum(Y[:, dictionary[diagnosis]]), diagnosis])

    hist.sort()
    hist = np.array(hist)

    plt.bar(range(len(hist[:,0])), hist[:,0])
    plt.xticks((range(len(hist[:,0]))), hist[:,1], rotation = 90, fontsize='xx-small')
    plt.show()


if __name__ == "__main__":
    xy = load_dataset(save_png=True)
    train_test_split_png()