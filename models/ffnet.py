import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader

class FourierLayer(nn.Module):
    def __init__(self, in_features, out_features, scale):
        super().__init__()
        B = torch.randn(in_features, out_features) * scale
        self.register_buffer("B", B)
    
    def forward(self, x):
        x_proj = torch.matmul(2 * math.pi * x, self.B)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out

class SignalRegressor(nn.Module):
    def __init__(self, in_features, fourier_features, hidden_features, hidden_layers, out_features, scale):
        super().__init__()
        self.net = []
        if fourier_features is not None:
            self.net.append(FourierLayer(in_features, fourier_features, scale))
            self.net.append(nn.Linear(2 * fourier_features, hidden_features))
            self.net.append(nn.ReLU())
        else:
            self.net.append(nn.Linear(in_features, hidden_features))
            self.net.append(nn.ReLU())
        
        for i in range(hidden_layers - 1):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(hidden_features, out_features))
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

class FFNet:
    def __init__(self, params):
        self.params = params
        self.device = params["device"]
        self.model = SignalRegressor(
            in_features=params["in_features"],
            fourier_features=params["fourier_features"],
            hidden_features=params["hidden_features"],
            hidden_layers=params["hidden_layers"],
            out_features=params["out_features"],
            scale=params["scale"]
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
            train_loss = F.mse_loss(output, train_values)
            train_loss.backward()
            optim.step()

            if not step % summary_interval:
                self.model.eval()
                with torch.no_grad():
                    prediction = self.model(test_coords)
                    test_loss = F.mse_loss(prediction, test_values)
                    print(f"Step: {step}, Test MSE: {test_loss.item():.6f}")
                    # Note: plot_data function call removed as it's not defined in this scope

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