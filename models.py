import sqlite3 as sql
import pandas as pd
import torch.nn as nn # type: ignore
import torch # type: ignore
import numpy as np
from torch.utils.data import Dataset # type: ignore
from torch.utils.data import DataLoader # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore
import tqdm # type: ignore  
from utils import *
import wandb # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore
import random


state_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
state_encoder.fit(np.array(list(state_dict.keys())).reshape(-1, 1))

industry_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
industry_encoder.fit(np.array([int(code) for code in naics_codes]).reshape(-1, 1))

from sklearn.preprocessing import MinMaxScaler

class EconDataset(Dataset):
    def __init__(self, data, state_col, industry_col, unemployment_col, response_col):
        self.data = data.copy()
        self.state_col = state_col
        self.industry_col = industry_col
        self.unemployment_col = unemployment_col
        self.response_col = response_col
        self._preprocess_data()

    def _preprocess_data(self):
        self.encoded_states = state_encoder.transform(self.data[[self.state_col]].values.reshape(-1, 1))
        self.encoded_industries = industry_encoder.transform(self.data[[self.industry_col]].values.reshape(-1, 1))

        # Normalize Unemployment Rate
        unemployment_scaler = MinMaxScaler()
        self.unemployment_stats = unemployment_scaler.fit_transform(self.data[[self.unemployment_col]].values.reshape(-1, 1))

        # Normalize Response Variable
        response_scaler = MinMaxScaler()
        self.response_scaler = response_scaler  # Store the scaler for later use
        self.normalized_responses = response_scaler.fit_transform(self.data[[self.response_col]].values.reshape(-1, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = self.encoded_states[idx]
        industry = self.encoded_industries[idx]
        unemployment = self.unemployment_stats[idx]
        predictor = np.concatenate((unemployment, state, industry), axis=0)
        response = self.normalized_responses[idx]
        return torch.tensor(predictor, dtype=torch.float32), torch.tensor(response, dtype=torch.float32).view(1)

    def denormalize_response(self, normalized_response):
        return self.response_scaler.inverse_transform(normalized_response.reshape(-1, 1))
    
# Base Model Architecture  
class SurvivalRateModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout, output_size=1):
        super(SurvivalRateModel, self).__init__()

        # Define layers:
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size),
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

    def train_loop(
        self,
        dataloader,
        num_epochs=100,
        learning_rate=0.01,
        device="cpu",
        suppress=False,
        val_dataloader=None,
        patience=40,
        optimizer=None,
        uses_wandb=False,
    ):
        self.to(device)
        criterion = nn.MSELoss()
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.train()
        best_val_loss = float("inf")
        epochs_no_improve = 0
        train_loss_cache = []
        val_loss_cache = []

        for epoch in tqdm.tqdm(range(num_epochs), desc="Training Epochs"):
            self.train()
            total_loss = 0
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                loss = self.train_step(x, y, criterion, optimizer)
                total_loss += loss
            train_loss = total_loss / len(dataloader)
            train_loss_cache.append(train_loss)
            if uses_wandb:
                wandb.log({"train_loss": train_loss}, step=epoch)

            if val_dataloader:
                self.eval()
                val_total_loss = 0
                with torch.no_grad():
                    for x_val, y_val in val_dataloader:
                        x_val, y_val = x_val.to(device), y_val.to(device)
                        outputs = self(x_val)
                        val_loss = criterion(outputs, y_val).item()
                        val_total_loss += val_loss
                val_loss = val_total_loss / len(val_dataloader)
                val_loss_cache.append(val_loss)
                if uses_wandb:
                    wandb.log({"val_loss": val_loss}, step=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_model_state = self.state_dict()  # Save best model
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if not suppress:
                            print(f"Early stopping at epoch {epoch + 1}")
                        self.load_state_dict(best_model_state) # Load best model
                        break

            if not suppress:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}"
                    + (f", Val Loss: {val_loss:.4f}" if val_dataloader else "")
                )

        return train_loss_cache, val_loss_cache

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
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size = 1, l1_lambda = 1e-2, l2_lambda = 1e-2, noise_std=0.05, dropout_rate=0.25):
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
    
# Try using embeddings to lower weights of categorical variables
class SurvivalRateModel_Embeddings(SurvivalRateModel):
    def __init__(self, num_states, num_industries, embedding_dim=8, hidden_size1=64, hidden_size2=32):
        # Calculate the input size for the main network
        combined_input_size = 1 + embedding_dim + embedding_dim  # unemployment + state_emb + industry_emb
        super().__init__(combined_input_size, hidden_size1, hidden_size2)
        
        # Create embeddings for categorical variables
        self.state_embedding = nn.Embedding(num_states, embedding_dim)
        self.industry_embedding = nn.Embedding(num_industries, embedding_dim)
    
    def forward(self, x):
        # Split the input
        unemployment = x[:, 0:1]  # First column
        state_oh = x[:, 1:52]     # Next 51 columns
        industry_oh = x[:, 52:]   # Remaining columns
        
        # Convert one-hot back to indices for embedding lookup
        state_idx = torch.argmax(state_oh, dim=1)
        industry_idx = torch.argmax(industry_oh, dim=1)
        
        # Get embeddings
        state_emb = self.state_embedding(state_idx)
        industry_emb = self.industry_embedding(industry_idx)
        
        # Combine features
        combined = torch.cat([unemployment, state_emb, industry_emb], dim=1)
        
        # Pass through layers
        return 100 * self.layers(combined)




def analyze_survival_rate(model, state, industry):
    """
    Demonstrates a few hardcoded sample predictions from the model,
    then plots how the predicted survival rate changes with unemployment rate
    for a specific (state, industry) configuration.
    """

    # Put model in evaluation mode
    model.eval()
    # 1) Collect a few random sample outputs
    
    # Generate random configurations
    random_configs = []
    for _ in range(3):
        unemp = round(random.uniform(2.0, 8.0), 1)  # Random unemployment between 2.0 and 8.0
        state_idx = random.choice(list(numerical_state_rev.keys()))
        industry_idx = random.choice(list(numerical_industry_rev.keys()))
        random_configs.append((unemp, state_idx, industry_idx))
    
    sample_results = []
    
    with torch.no_grad():
        for (unemp, state_idx, industry_idx) in random_configs:
            # Build input tensor
            sample_input = torch.tensor(np.concatenate([
                np.array([unemp]).reshape(-1, 1),
                state_encoder.transform([[state_idx]]).reshape(-1, 1),
                industry_encoder.transform([[industry_idx]]).reshape(-1, 1)
            ], axis=0), dtype=torch.float32).T
            
            output = model(sample_input)
            # Include actual state and industry names
            sample_results.append((
                unemp, 
                numerical_state_rev[state_idx],  # Convert index to state name
                numerical_industry_rev[industry_idx],  # Convert index to industry name
                output.item()
            ))

    # Print the hardcoded sample outputs
    print("Sample Outputs:")
    for res in sample_results:
        print(f"Unemployment={res[0]}, State={res[1]}, Industry={res[2]} => Survival Prediction={res[3]:.4f}")

    # 2) Plot unemployment rate vs survival rate for one configuration
    # Choose a specific state and industry
    chosen_state = state
    chosen_industry = industry

    unemployment_values = np.linspace(0, 10, 1100)  # from 0 to 10
    predictions = []

    with torch.no_grad():
        for unemp in unemployment_values:
            sample_input = torch.tensor(np.concatenate([
                np.array([unemp]).reshape(-1, 1),
                state_encoder.transform([[chosen_state]]).reshape(-1, 1),
                industry_encoder.transform([[chosen_industry]]).reshape(-1, 1)
            ], axis=0), dtype=torch.float32).T
            output = model(sample_input).item()
            predictions.append(output)

    # Plotting
    plt.figure(figsize=(6, 4))
    plt.plot(unemployment_values, predictions, marker='o', markersize=0.4)
    plt.title(f"Survival Rate vs. Unemployment (State={numerical_state_rev[chosen_state]}, Industry={numerical_industry_rev[chosen_industry]})")
    plt.xlabel("Unemployment Rate")
    plt.ylabel("Predicted Survival Rate")
    plt.grid(True)
    plt.show()

def analyze_model_performance(model, train_loss, val_loss, train_dataloader, test_dataloader, device="cpu"):
    """
    Plots training and validation loss curves, then computes and prints MSE and R^2
    for both training and validation sets.
    """

    # 1) Plot training and validation loss
    # Extend the validation loss to match the length of the training loss
    if len(val_loss) < len(train_loss):
        val_loss_extended = np.interp(np.arange(len(train_loss)), np.linspace(0, len(train_loss) - 1, len(val_loss)), val_loss)
    else:
        val_loss_extended = val_loss

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss', linewidth=0.4, alpha=0.8)
    plt.plot(val_loss_extended, label='Validation Loss', linewidth=0.4, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

    # 2) Compute additional metrics on training and validation sets
    model.eval()
    train_preds, train_actuals = [], []
    test_preds, test_actuals = [], []

    with torch.no_grad():
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x).cpu().numpy()
            train_preds.append(outputs)
            train_actuals.append(y.cpu().numpy())

        for x_test, y_test in test_dataloader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            test_outputs = model(x_test).cpu().numpy()
            test_preds.append(test_outputs)
            test_actuals.append(y_test.cpu().numpy())

    # Concatenate predictions and actuals
    train_preds = np.concatenate(train_preds, axis=0)
    train_actuals = np.concatenate(train_actuals, axis=0)
    test_preds = np.concatenate(test_preds, axis=0)
    test_actuals = np.concatenate(test_actuals, axis=0)

    # Calculate mean squared error and R^2
    train_mse = mean_squared_error(train_actuals, train_preds)
    test_mse = mean_squared_error(test_actuals, test_preds)
    train_r2 = r2_score(train_actuals, train_preds)
    test_r2 = r2_score(test_actuals, test_preds)

    print(f"Training MSE: {train_mse:.4f}, R^2: {train_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}, R^2: {test_r2:.4f}")