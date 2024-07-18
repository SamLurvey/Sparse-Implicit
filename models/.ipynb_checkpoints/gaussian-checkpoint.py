import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class GaussianLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.normal_(0, 1 / self.in_features)
            else:
                self.linear.weight.normal_(0, np.sqrt(2 / self.in_features))
        
    def forward(self, input):
        return torch.exp(-(self.linear(input)**2) / (2 * self.sigma**2))

class GaussianRegressor(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True):
        super().__init__()
        self.net = []
        self.net.append(GaussianLayer(in_features, hidden_features, is_first=True))
        
        for i in range(hidden_layers-1):
            self.net.append(GaussianLayer(hidden_features, hidden_features, is_first=False))
        
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.normal_(0, np.sqrt(2 / hidden_features))
                
            self.net.append(final_linear)
        else:
            self.net.append(GaussianLayer(hidden_features, out_features, is_first=False))
        
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)

class ArrayDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.arr = data

    def __getitem__(self, idx):
        tensor_arr = torch.from_numpy(self.arr).float()
        h_axis = torch.linspace(0, 1, steps=self.arr.shape[0])
        w_axis = torch.linspace(0, 1, steps=self.arr.shape[1])
        d_axis = torch.linspace(0, 1, steps=self.arr.shape[2])
        grid = torch.stack(torch.meshgrid(h_axis, w_axis, d_axis, indexing='ij'), dim=-1)
        return grid, tensor_arr

    def __len__(self):
        return 1

class Gaussian:
    def __init__(self, params):
        self.params = params
        self.device = params["device"]
        self.model = GaussianRegressor(
            in_features=params["in_features"],
            hidden_features=params["hidden_features"],
            hidden_layers=params["hidden_layers"],
            out_features=params["out_features"],
            outermost_linear=params.get("outermost_linear", True)
        ).to(self.device)

    def train(self, data, total_steps=1000, summary_interval=100):
        array_loader = self.create_loader(data)
        grid, array = next(iter(array_loader))
        grid, array = grid.squeeze().to(self.device), array.squeeze().to(self.device)
        train_coords, train_values = grid.reshape(-1, 3), array.reshape(-1, 1)
        test_coords, test_values = grid.reshape(-1, 3), array.reshape(-1, 1)

        optim = torch.optim.Adam(lr=self.params["lr"], params=self.model.parameters())

        for step in range(1, total_steps + 1):
            self.model.train()
            optim.zero_grad()
            output = self.model(train_coords)
            train_loss = torch.nn.functional.mse_loss(output, train_values)
            train_loss.backward()
            optim.step()

            if not step % summary_interval:
                self.model.eval()
                with torch.no_grad():
                    prediction = self.model(test_coords)
                    test_loss = torch.nn.functional.mse_loss(prediction, test_values)
                    print(f"Step: {step}, Test MSE: {test_loss.item():.6f}")

        return test_loss.item()

    def predict(self, coords):
        self.model.eval()
        with torch.no_grad():
            return self.model(coords)

    @staticmethod
    def create_loader(data):
        array_data = ArrayDataset(data)
        return DataLoader(array_data, batch_size=1)

    def get_compression_ratio(self, original_size):
        model_size = sum(p.numel() for p in self.model.parameters())
        return original_size / model_size