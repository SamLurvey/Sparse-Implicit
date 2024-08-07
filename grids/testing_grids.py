import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import itertools
import time

from plotting import plot_data
import data_loading
import os
from pathlib import Path

import numpy as np

def reshape_data(data, target_shape):
    """
    Reshape the data to the target shape by selecting every other point along reduced dimensions.
    
    :param data: numpy array of shape (192, 249, 16)
    :param target_shape: tuple of the desired shape
    :return: reshaped numpy array
    """
    original_shape = (192, 249, 16)
    
    if data.shape != original_shape:
        raise ValueError(f"Input data must have shape {original_shape}")

    if original_shape == target_shape:
        return data
    
    if target_shape not in [(192, 125, 16), (192, 249, 8), (96, 125, 16), (96, 125, 8)]:
        raise ValueError("Target shape must be one of (192, 125, 16), (192, 249, 8), (96, 125, 16), or (96, 125, 8)")

    result = data.copy()

    # Reshape first dimension (192 -> 96)
    if target_shape[0] == 96:
        result = result[::2, :, :]

    # Reshape second dimension (249 -> 125)
    if target_shape[1] == 125:
        result = result[:, ::2, :]

    # Reshape third dimension (16 -> 8)
    if target_shape[2] == 8:
        result = result[:, :, ::2]

    return result

def evaluate_model(model, original_data, transformed_data, max_iters, grid_size):
    start_time = time.time()
    
    # Train the model on transformed data
    transformed_data = reshape_data(transformed_data,grid_size)
    mse_transformed = model.train(transformed_data, total_steps = max_iters)
    training_time = time.time() - start_time
    
    # Test the model on the same data
    array_loader = model.create_loader(original_data)
    grid, array = next(iter(array_loader))
    grid, array = grid.squeeze().to(model.device), array.squeeze().to(model.device)
    coords, values = grid.reshape(-1, 3), array.reshape(-1, 1)
    prediction_transformed = model.predict(coords)
    
    # Reshape prediction to original shape
    original_shape = array.cpu().numpy().shape
    predicted_array_transformed = prediction_transformed.reshape(original_shape).cpu().numpy()
    
    # Inverse transform the predictions
    prediction_original = data_loading.reverse_transform(predicted_array_transformed)

    # Calculate MSE on original (non-log) scale
    mse_original = np.mean((original_data - prediction_original)**2)
    
    # Plot the original and predicted data
    #plot_data(original_data, prediction_original)
    #plot_data(transformed_data, predicted_array_transformed)
    
    # Compute compression ratio
    compression_ratio = model.get_compression_ratio(original_data.size)
    
    return mse_original, mse_transformed, compression_ratio, training_time, prediction_original

def evaluate_models(hyperparameters, original_samples, transformed_samples, file_names, model_classes):
    results = []
    
    # Create a directory to store the predictions
    predictions_dir = Path("saved_predictions")
    predictions_dir.mkdir(exist_ok=True)
    
    for i, (original_data, transformed_data, file_name) in enumerate(zip(original_samples, transformed_samples, file_names)):
        print(f"\nEvaluating on data sample {i+1}: {file_name}")
        
        for ModelClass in model_classes:
            model = ModelClass(hyperparameters)
            
            print(f"\nEvaluating {ModelClass.__name__}")
            print("Hyperparameters:")
            for key, value in hyperparameters.items():
                print(f"  {key}: {value}")
            
            mse_original, mse_transformed, compression_ratio, training_time, prediction_original = evaluate_model(model, original_data, transformed_data, hyperparameters["max_iters"], hyperparameters["grid_size"])
            
            print(f"MSE (original scale): {mse_original}")
            print(f"MSE (transformed scale): {mse_transformed}")
            print(f"Compression Ratio: {compression_ratio}")
            print(f"Training time: {training_time}")
            
            # Save the prediction
            if i == 0:
                prediction_filename = f"{ModelClass.__name__}_sample{i+1}_grid{hyperparameters['grid_size']}"
                prediction_filename += ".npy"
                prediction_path = predictions_dir / prediction_filename
                np.save(prediction_path, prediction_original)
            
                results.append({
                    'model': ModelClass.__name__,
                    'data_sample': i+1,
                    'file_name': file_name,
                    'original_mse': mse_original,
                    'compression_ratio': compression_ratio,
                    'transformed_mse': mse_transformed,
                    'training_time': training_time,
                    'prediction_path': str(prediction_path),
                    **hyperparameters
                })
            else:
                results.append({
                    'model': ModelClass.__name__,
                    'data_sample': i+1,
                    'file_name': file_name,
                    'original_mse': mse_original,
                    'compression_ratio': compression_ratio,
                    'transformed_mse': mse_transformed,
                    'training_time': training_time,
                    **hyperparameters
                })                
    
    return results

def hyperparameter_grid_search(base_grid, model_grids):
    all_results = []
    
    # Load 5 random data samples
    original_samples, file_names = data_loading.load_first_five_data()

    # Log transform data
    transformed_samples = []
    for sample in original_samples:
        _, transformed = data_loading.transform_data(sample)
        transformed_samples.append(transformed)

    for ModelClass, grid in model_grids.items():
        hyperparameter_combinations = [dict(zip(grid.keys(), values)) 
                                       for values in itertools.product(*grid.values())]
        
        for hyperparameters in hyperparameter_combinations:
            results = evaluate_models(hyperparameters, original_samples, transformed_samples, file_names, [ModelClass])
            all_results.extend(results)
    
    results_df = pd.DataFrame(all_results)
    
    # Save results to CSV
    results_df.to_csv('model_comparison_results_with_predictions.csv', index=False)
    
    return results_df


if __name__ == "__main__":
    results = hyperparameter_grid_search(base_grid, model_grids)
    print("\nFinal Results:")
    print(results)