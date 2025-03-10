import sqlite3 as sql
import pandas as pd
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
import tqdm



class EconDataset(Dataset):
    def __init__(self, data, state_col, industry_col, unemployment_col, response_col):
        self.data = data.copy()
        self.state_col = state_col
        self.industry_col = industry_col
        self.unemployment_col = unemployment_col
        self.response_col = response_col
        self.state_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.industry_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self._preprocess_data()

    def _preprocess_data(self):
        self.encoded_states = self.state_encoder.fit_transform(self.data[[self.state_col]].values.reshape(-1, 1)) # Applying one-hot encoding to the state column
        self.encoded_industries = self.industry_encoder.fit_transform(self.data[[self.industry_col]].values.reshape(-1, 1)) # Applying one-hot encoding to the industry
        self.unemployment_stats = self.data[self.unemployment_col].values.reshape(-1, 1) # Turns into column vector
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = self.encoded_states[idx]
        industry = self.encoded_industries[idx]
        unemployment = self.unemployment_stats[idx]
        predictor = np.concatenate((unemployment, state, industry), axis=0)
        response = self.data[self.response_col].values[idx]
        return torch.tensor(predictor, dtype=torch.float32), torch.tensor(response, dtype=torch.float32).view(1) #Response is reshaped to a column vector.
    
# Base Model Architecture  
class SurvivalRateModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size = 1):
        super(SurvivalRateModel, self).__init__()

        # Define layers:
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return 100 * self.layers(x)

    def train_step(self, x, y, criterion, optimizer):
        optimizer.zero_grad()
        outputs = self(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()
        return loss.item()
    
    def train_loop(self, dataloader, num_epochs=100, learning_rate=0.01, device="cpu", suppress=False, val_dataloader=None, patience = 40):
        self.to(device)
        criterion = nn.MSELoss()
        loss_cache = []
        val_loss_cache = []
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.train() #set model to train mode.
        for epoch in tqdm.tqdm(range(num_epochs), desc="Training Epochs"):
            # Training phase
            self.train()
            total_loss = 0
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                loss = self.train_step(x, y, criterion, optimizer)
                loss_cache.append(loss)
                total_loss += loss
            train_loss = total_loss/len(dataloader)
            
            # Validation phase (if validation data is provided)
            if val_dataloader is not None:
                self.eval()
                val_total_loss = 0
                with torch.no_grad():
                    for x_val, y_val in val_dataloader:
                        x_val, y_val = x_val.to(device), y_val.to(device)
                        outputs = self(x_val)
                        val_loss = criterion(outputs, y_val).item()
                        val_total_loss += val_loss
                val_loss = val_total_loss/len(val_dataloader)
                val_loss_cache.append(val_loss)


                best_val_loss = float('inf')
                # Stops model early if validation loss does not improve for 'patience' epochs
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    # Save the best model (as shown in the previous example)
                    best_model_state = self.state_dict() # crucial to copy here
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        print(f'Early stopping triggered after epoch {epoch+1}')
                        self.load_state_dict(best_model_state) # restore best model
                        break
                
                if not suppress:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            elif not suppress:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')
        
        if val_dataloader is not None:
            return loss_cache, val_loss_cache
        return loss_cache
    
    def evaluate(self, dataloader, device="cpu"):
        self.to(device)
        self.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                outputs = self(x)
                predictions.append(outputs.cpu().numpy())
                actuals.append(y.cpu().numpy())
        return (np.concatenate(predictions), np.concatenate(actuals))
    
    def predict(self, x, device="cpu"):
        self.to(device)
        self.eval()
        with torch.no_grad():
            x = x.to(device)
            return self(x).cpu().numpy()

#Implements batchnorm, dropout, elastic net and noise injection
class SurvivalRateModel_AggressiveReg(SurvivalRateModel):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size = 1, l1_lambda = 0.0005, l2_lambda = 0.01, noise_std=0.05, dropout_rate=0.2):
        super(SurvivalRateModel_AggressiveReg, self).__init__(input_size, hidden_size1, hidden_size2, output_size)
        
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate
        
        # Override the layers with more regularization
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size),
            nn.Sigmoid()
        )

    def train_step(self, x, y, criterion, optimizer):
        optimizer.zero_grad()

        # Add noise to the input data to improve generalization
        noise = torch.zeros_like(x)
        noise[0] = torch.randn_like(x[0]) * self.noise_std
        x_noisy = x + noise

        outputs = self(x_noisy)
        loss = criterion(outputs, y)

        # Add L1 and L2 regularization (Elastic Net) to reduce overfitting
        l1_reg = torch.tensor(0., device=x.device)
        l2_reg = torch.tensor(0., device=x.device)
        for param in self.parameters():
            l1_reg += torch.norm(param, 1)
            l2_reg += torch.norm(param, 2)

        loss = loss + self.l1_lambda * l1_reg + self.l2_lambda * l2_reg

        loss.backward()
        optimizer.step()
        return loss.item()

#Implements additional layers, and leaky relu activation function
class SurvivalRateModel_DeepLearning(SurvivalRateModel):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size = 1):
        # We call the parent constructor but we'll override the layers
        super(SurvivalRateModel_DeepLearning, self).__init__(input_size, hidden_size1, hidden_size2, output_size)

        # Override the layers with a deeper architecture
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size2, hidden_size3),
            nn.BatchNorm1d(hidden_size3),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(hidden_size3, hidden_size4),
            nn.BatchNorm1d(hidden_size4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size4, output_size),
            nn.Sigmoid()
        )

#Implements batchnorm, dropout, elastic net and noise injection with strong regularization
class SurvivalRateModel_Combined(SurvivalRateModel):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size = 1, l1_lambda = 0.0005, l2_lambda = 0.01, noise_std=0.05, dropout_rate=0.2):
        # We call the parent constructor but we'll override the layers
        super(SurvivalRateModel_Combined, self).__init__(input_size, hidden_size1, hidden_size2, output_size)
        
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate

        # Override the layers with a deeper architecture and more regularization
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_size2, hidden_size3),
            nn.BatchNorm1d(hidden_size3),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_size3, hidden_size4),
            nn.BatchNorm1d(hidden_size4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size4, output_size),
            nn.Sigmoid()
        )
        

    def train_step(self, x, y, criterion, optimizer):
        optimizer.zero_grad()

        # Add noise to the input data to improve generalization
        noise = torch.zeros_like(x)
        noise[0] = torch.randn_like(x[0]) * self.noise_std
        x_noisy = x + noise

        outputs = self(x_noisy)
        loss = criterion(outputs, y)

        # Add L1 and L2 regularization (Elastic Net) to reduce overfitting
        l1_reg = torch.tensor(0., device=x.device)
        l2_reg = torch.tensor(0., device=x.device)
        for param in self.parameters():
            l1_reg += torch.norm(param, 1)
            l2_reg += torch.norm(param, 2)

        loss = loss + self.l1_lambda * l1_reg + self.l2_lambda * l2_reg

        loss.backward()
        optimizer.step()
        return loss.item()
