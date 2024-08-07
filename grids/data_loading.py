import numpy as np
import os


def load_dataset_info():
    base_path = "../outer"
    train_files = [line.strip() for line in open(os.path.join(base_path, "train.txt"))]
    test_files = [line.strip() for line in open(os.path.join(base_path, "test.txt"))]
    
    all_files = []
    for file_path in train_files + test_files:
        full_path = os.path.join(base_path, file_path)
        if os.path.isfile(full_path) and full_path.endswith('.npy'):
            all_files.append(full_path)
    
    print(f"Total files found: {len(all_files)}")
    return all_files

def load_random_data(num_samples=5):
    all_files = load_dataset_info()
    selected_files = random.sample(all_files, num_samples)
    data_samples = [np.load(file).astype('int16') for file in selected_files]
    return data_samples, selected_files

def load_first_five_data():
    base_path = "../outer"
    train_files = [line.strip() for line in open(os.path.join(base_path, "train.txt"))]
    test_files = [line.strip() for line in open(os.path.join(base_path, "test.txt"))]
    
    all_files = []
    for file_path in train_files + test_files:
        full_path = os.path.join(base_path, file_path)
        if os.path.isfile(full_path) and full_path.endswith('.npy'):
            all_files.append(full_path)
            if len(all_files) == 50:
                break

    data_samples = [np.load(file).astype('int16') for file in all_files]
    return data_samples, all_files

def transform_data(data):
    # Create a copy of the input array as float
    result = data.astype(float)
    
    # Create a mask for values greater than 64
    valid_mask = result > 64
    
    # Apply the transformation only to valid elements
    transformed = result.copy()
    transformed[valid_mask] = np.log(transformed[valid_mask] - 64) / 6
    
    return result, transformed

def reverse_transform(data):
    result = np.exp(data * 6) + 64
    result[result < 67] = 0  # Set values below 64 to 0
    result = result.astype(int)
    return result