import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from one_hot_encoding import encode_one_cdr3
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


# create train data
normal_CDR3 = pd.read_csv("NormalCDR3.txt", header=None)
cancer_CDR3 = pd.read_csv("TumorCDR3.txt", header=None)
train_CDR3 = pd.concat([normal_CDR3, cancer_CDR3], ignore_index=True)
train_CDR3['label'] = [0] * len(normal_CDR3) + [1] * len(cancer_CDR3)
train_CDR3.columns = ['sequence', 'label']

# create test data
normal_CDR3_test = pd.read_csv("NormalCDR3_test.txt", header=None)
cancer_CDR3_test = pd.read_csv("TumorCDR3_test.txt", header=None)
test_CDR3 = pd.concat([normal_CDR3_test, cancer_CDR3_test], ignore_index=True)
test_CDR3['label'] = [0] * len(normal_CDR3_test) + [1] * len(cancer_CDR3_test)
test_CDR3.columns = ['sequence', 'label']

# get AA for AA index
aa_index = pd.read_csv("AAidx_PCA.txt", sep='\t')
total_aa = aa_index.iloc[:, 0].tolist()


# # Encode Training and Testing Data
# representing sequence as a matrix for cnn
train_data_encoded = np.array([encode_one_cdr3(seq, total_aa) for seq in train_CDR3['sequence']])
test_data_encoded = np.array([encode_one_cdr3(seq, total_aa) for seq in test_CDR3['sequence']])

# build CNN mode;
def build_model(input_data):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_data.shape[1],)),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    return model


# Step 5: Train the Model
model = build_model(train_data_encoded)

x_train_partial, x_val, y_train_partial, y_val = train_test_split(
    train_data_encoded, 
    train_CDR3['label'].values, 
    test_size=35000, 
    random_state=123,  # For reproducibility
    shuffle=True,      # Ensures shuffling
    stratify=train_CDR3['label'].values  # Maintain class balance
)

# Model training
history = model.fit(
    x_train_partial,
    y_train_partial,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)


# Evaluate the Model
results = model.evaluate(test_data_encoded, test_CDR3['label'].values)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Visualization of Training
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
