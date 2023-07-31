import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

"""## downloading the dataset"""

#!curl -O https://raw.githubusercontent.com/FSchierok/ml-seminar/main/students.csv

df = pd.read_csv("students.csv")

"""# preparing the data"""

df = shuffle(df, random_state=42)
target = (
    df[["math score", "reading score", "writing score"]]
) / 100  # scale between 0 and 1
data = df.drop(
    columns=["Unnamed: 0", "math score", "reading score", "writing score"]
)  # Unnamed is the pandas index

"""## encoding the features
a higher number means a higher score is to be expected
look below for the analysis
"""

gender_d = {"male": 1, "female": 2}
race_d = {"group A": 1, "group B": 2, "group C": 3, "group D": 4, "group E": 5}
parents_d = {
    "some high school": 1,
    "some college": 3,
    "high school": 2,
    "associate's degree": 4,
    "bachelor's degree": 5,
    "master's degree": 6,
}
lunch_d = {"free/reduced": 1, "standard": 2}
course_d = {"none": 1, "completed": 2}


def encode(df):
    df["gender"] = df["gender"].map(gender_d)
    df["lunch"] = df["lunch"].map(lunch_d)
    df["parental level of education"] = df["parental level of education"].map(parents_d)
    df["race/ethnicity"] = df["race/ethnicity"].map(race_d)
    df["test preparation course"] = df["test preparation course"].map(course_d)
    return df


"""## encoding"""

data = encode(data)
target = target.to_numpy()

"""## create a feature

"""

from scipy.special import expit

data["math talent"] = expit(np.random.normal(target[:, 0] * np.sum(data, axis=1), 1))
data["reading talent"] = expit(np.random.normal(target[:, 1] * np.sum(data, axis=1), 1))
data["writing talent"] = expit(np.random.normal(target[:, 2] * np.sum(data, axis=1), 1))
data = data.to_numpy()

"""## splitting
10k validation, 10k testing, 20k training
"""

X_nn, X_test, Y_nn, Y_test = train_test_split(
    data, target, test_size=0.25, random_state=42
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_nn, Y_nn, test_size=0.33, random_state=42
)

# Hyperspace
param_space = {
    "hidden_layers": [1, 2, 3, 4, 5],
    "learning_rate": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "nodes_first_layer": [16, 32, 64, 128],
    "activation": ["selu", "sigmoid", "tanh"],
}
import itertools

value_combis = itertools.product(*[v for v in param_space.values()])
param_combis = [
    {key: value for key, value in zip(param_space.keys(), combi)}
    for combi in value_combis
]

hyper_results = list()
losses = list()
"""# Network setup"""

from keras import Sequential
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint

from tqdm import tqdm

print(f"total combis: {len(param_combis)}, predicted runtime: {len(param_combis)*20}s")
for idx, params in enumerate(tqdm(param_combis)):
    model = Sequential()
    model.add(Input(shape=X_train.shape[1]))
    for i in range(params["hidden_layers"]):
        model.add(
            Dense(
                units=params["nodes_first_layer"] // (i + 1),
                activation=params["activation"],
            )
        )
    model.add(Dense(units=Y_train.shape[1], activation="selu"))
    model.compile(
        loss="MSE",
        optimizer=keras.optimizers.AdamW(learning_rate=params["learning_rate"]),
        metrics=["cosine_similarity"],
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,  # number of epochs with no improvement
        restore_best_weights=True,
    )
    string_config = ""
    for key, value in params.items():
        string_config += key + "=" + str(value)
    filepath = f"checkpoints/{string_config}.keras"
    checkpoint = ModelCheckpoint(
        filepath, monitor="val_loss", verbose=0, save_best_only=True, mode="min"
    )
    hist = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=300,
        batch_size=512,
        verbose=0,
        callbacks=[early_stopping, checkpoint],
    )
    hyper_results.append(
        {
            **params,
            "epoch": int(np.argmin(hist.history["val_loss"])),
            "val_loss": float(np.min(hist.history["val_loss"])),
        }
    )
    loss, cosine = model.evaluate(X_test, Y_test, verbose=0)
    hyper_results.append({**params, "val_loss": float(loss), "cos": float(cosine)})
    losses.append(loss)

best_model = np.argmin(losses)
print(f"best model: #{best_model}")
print(hyper_results[best_model])

import json

with open("hyperV2.json", "w") as j:
    json.dump(hyper_results, j)
