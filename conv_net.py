import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import (
    TensorBoard
)
from keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Conv2D,
    MaxPooling2D,
    UpSampling2D, Input, Flatten, Reshape, Dropout
)
from keras.losses import (
    mean_squared_error, binary_crossentropy
)
from keras.models import (
    Sequential, load_model, Model
)
from keras import optimizers
import pickle as pkl
import os
from dataset import get_number_of_diagnosis, get_statistic_of_diagnosis, load_dataset
from generator import generator_png
from sklearn.model_selection import train_test_split


num_cores = 64
kernel = 3
num_latent = 60

pkl_weights = "C:\\Users\\donte_000\\Downloads\\ecg_1078\\weights.pkl"
pkl_filename = "C:\\Users\\donte_000\\Downloads\\ecg_1078\\dataset_fixed_baseline.pkl"
raw_dataset = "C:\\Users\\donte_000\\Downloads\\ecg_1078\\data_1078.json"

train_folder_path = "C:\\Users\\donte_000\\Downloads\\ecg_1078\\train_png"
test_folder_path = "C:\\Users\\donte_000\\Downloads\\ecg_1078\\test_png"


num_leads_signal = 1
win_len = 208
batch_size = 20
#shape : [num_patient, num lead, y_axis, x_axis]
if os.path.exists(pkl_filename):
    infile = open(pkl_filename, 'rb')
    dataset = pkl.load(infile)
    infile.close()
else:
    print("Pkl_file is not found.")
    dataset = load_dataset(raw_dataset)

X, Y = dataset["x"], dataset["y"]

#xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.33, random_state=42)

train_generator = generator_png(X_png_folder_path=train_folder_path, Y=Y, win_len=win_len, batch_size=batch_size, num_leads_signal=num_leads_signal)
test_generator = generator_png(X_png_folder_path=test_folder_path, Y=Y, win_len=win_len, batch_size=300, num_leads_signal=num_leads_signal)
#train_generator = generator_png(X=xtrain, Y=ytrain, batch_size=batch_size, diagnosis=189, num_leads_signal=num_leads_signal)
mmm = next(train_generator)
num_diag = 212

model = Sequential()
model.add(Conv2D(num_cores, kernel_size=kernel,
                 activation="relu",
                 input_shape=(win_len, 145, num_leads_signal), padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv2D(num_cores, kernel_size=kernel, activation="relu", padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv2D(num_cores, kernel_size=kernel, activation="relu", padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
#model.add(Conv2D(num_cores, kernel_size=kernel, activation=K.elu, padding='same'))
#model.add(MaxPooling2D(pool_size=2))
#model.add(Dropout(0.3))
model.add(Reshape((win_len//8, -1)))
#model.add(Flatten())
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.3))
#model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(num_diag, activation='sigmoid'))


model.summary()
#sgd = optimizers.SGD(lr=0.001, decay=1e-6)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])

test_set = next(test_generator)

model.fit_generator(train_generator,
                    epochs=15,
                    steps_per_epoch=10,
                    validation_data=(test_set[0], test_set[1]))

model.save('model_bin'+'.h5')


#pred_test = model.predict(test_set[0])
#print(pred_test[0])
#true = test_set[1]
#print(true[0])