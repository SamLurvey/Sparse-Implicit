import torch

def calculate_initial_weights(data):
    data_tensor = torch.from_numpy(data).float()
    weights = torch.where(data_tensor != 0, torch.abs(data_tensor), torch.tensor(0.1))
    return weights / weights.sum()

def sample_data(data, weights, num_samples):
    # Flatten data and weights for sampling
    data_flat = data.view(-1)
    weights_flat = weights.view(-1)
    
    # Sample indices based on weights
    indices = torch.multinomial(weights_flat, num_samples, replacement=True)
    
    # Retrieve sampled data points
    sampled_data = data_flat[indices]
    return sampled_data, indices