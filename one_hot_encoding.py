import numpy as np

def encode_one_cdr3(cdr3, total_aa, length_cutoff=12):
    padded_sequence = cdr3.ljust(length_cutoff, 'X')[:length_cutoff]

    # Initialize a zero matrix of size (20, length_cutoff)
    one_hot_matrix = np.zeros((20, length_cutoff), dtype=int)

    for i, aa in enumerate(padded_sequence):
        if aa in total_aa:  # If the amino acid is in our mapping
            aa_idx = total_aa.index(aa)
            one_hot_matrix[aa_idx][i] = 1

    # Flatten the matrix into a single-dimensional array
    return one_hot_matrix.flatten()


def enhanced_encode_one_cdr3(cdr3, total_aa, pca_data, length_cutoff=12):
    # One-hot encoding
    one_hot_encoded = np.zeros((len(total_aa), length_cutoff), dtype=int)
    pca_features = []

    cdr3 = list(cdr3.ljust(length_cutoff, 'X'))[:length_cutoff]  # Pad/truncate sequence
    for idx, aa in enumerate(cdr3):
        if aa in total_aa:
            aa_index = total_aa.index(aa)
            one_hot_encoded[aa_index, idx] = 1
            pca_features.append(pca_data.loc[aa].values)
        else:
            pca_features.append(np.zeros(pca_data.shape[1]))  # Placeholder for unknown AA

    # Calculate statistics for PCA values
    pca_features = np.array(pca_features)
    pca_mean = pca_features.mean(axis=0)
    pca_max = pca_features.max(axis=0)
    
    # Flatten one-hot and concatenate PCA statistics
    return np.concatenate([one_hot_encoded.flatten(), pca_mean, pca_max])