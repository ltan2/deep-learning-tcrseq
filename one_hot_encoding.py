import numpy as np

def encode_one_cdr3(cdr3, total_aa, length_cutoff=12):
    encoded = np.zeros((len(total_aa), length_cutoff), dtype=int)
    
    cdr3 = list(cdr3.ljust(length_cutoff, 'X'))[:length_cutoff] # pad or truncate seq to specific lenght
    
    for idx, aa in enumerate(cdr3):
        if aa in total_aa:
            aa_index = total_aa.index(aa)
            encoded[aa_index, idx] = 1  # Set 1 where the amino acid matches
            
    return encoded.flatten()  # Return a flattened version to make it ready for a model