import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

amino_acid_to_index = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
labels_str = {
    'Benign': 0, 'Likely benign': 0,
    'Likely pathogenic': 1, 'Pathogenic': 1}
data_df = pd.read_pickle('extracted_data_W10.pkl')
real_acids = data_df['real_acid'].to_numpy()
var_acids = data_df['variant_acid'].to_numpy()

# Perform one-hot encoding for both real and variant acids
one_hot_encoded_real, one_hot_encoded_var = [], []
for real_acid, var_acid in zip(real_acids, var_acids):
    # One-hot encoding for real acid
    one_hot_vector_real = np.zeros(20)  # Create a zero vector of length 20
    index_real = amino_acid_to_index.get(real_acid, -1)  # Get the index of the real amino acid from the dictionary
    if index_real != -1:
        one_hot_vector_real[index_real] = 1  # Set the element at the index to 1
    one_hot_encoded_real.append(one_hot_vector_real)
    # One-hot encoding for variant acid
    one_hot_vector_var = np.zeros(20)  # Create a zero vector of length 20
    index_var = amino_acid_to_index.get(var_acid, -1)  # Get the index of the variant amino acid from the dictionary
    if index_var != -1:
        one_hot_vector_var[index_var] = 1  # Set the element at the index to 1
    one_hot_encoded_var.append(one_hot_vector_var)

# Convert the lists of one-hot encoded vectors into numpy arrays
ohe_real_acid_arr = np.array(one_hot_encoded_real)
ohe_var_acid_arr = np.array(one_hot_encoded_var)
# Contains OHE of real & variant acid, position, esm1b score, secondary structure frac., and LLRs of variant location
model_feature_arr = np.concatenate([ohe_real_acid_arr, ohe_var_acid_arr,
                                          data_df['position'].to_numpy().reshape(-1, 1),
                                          data_df['esm1b_score'].to_numpy().reshape(-1, 1),
                                          np.vstack([arr for arr in data_df['pred_2nd_struct_arr'].to_numpy()]),
                                          np.vstack([arr for arr in data_df['variant_loc_LLR_arr'].to_numpy()])],
                                          axis=-1)
scaler = MinMaxScaler()
scaled_model_feature_arr = scaler.fit_transform(model_feature_arr)

labels_encoded = np.array(list(map(lambda x: labels_str[x], data_df['label'].to_numpy())))
ohe_labels = np.eye(2)[labels_encoded]

seq_arr = np.empty((0, 10))
for p_seq in data_df['seq_arr'].to_numpy():
    protein_sequence_indices = [amino_acid_to_index[aa] for aa in p_seq]
    seq_arr = np.vstack([seq_arr, protein_sequence_indices])

input_layer = keras.layers.Input(shape=(65,))
seq_embedding_input = keras.layers.Input(shape=(10,))
seq_embedding = keras.layers.Embedding(20, 1, input_length=10)(seq_embedding_input)
seq_embedding_flat = keras.layers.Flatten()(seq_embedding)
combined_input = keras.layers.Concatenate()([input_layer, seq_embedding_flat])
hidden = keras.layers.Dense(64, activation='relu')(combined_input)
hidden = keras.layers.Dense(64, activation='relu')(hidden)
hidden = keras.layers.Dropout(0.5)(hidden)
hidden = keras.layers.Dense(32, activation='relu')(hidden)
hidden = keras.layers.Dense(32, activation='relu')(hidden)
output_layer = keras.layers.Dense(2, activation='softmax')(hidden)
model_v1 = keras.models.Model(inputs=[input_layer, seq_embedding_input], outputs=output_layer)
model_v1.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model_v1.fit([scaled_model_feature_arr, seq_arr], ohe_labels, batch_size=256, epochs=32,  validation_split=0.1, shuffle=True)
