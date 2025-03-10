from sklearn.metrics import mean_squared_error, r2_score # type: ignore
from models import state_encoder, industry_encoder
import torch # type: ignore
import numpy as np
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
import random

def generate_state_predictions(model, unemployment_rate, industry_code):
    """
    Generate predictions for each state based on the given model and encoders.

    Parameters:
    model (torch.nn.Module): The trained model for prediction.
    state_encoder (OneHotEncoder): Encoder for state names.
    industry_encoder (OneHotEncoder): Encoder for industry codes.
    state_dict (dict): Dictionary mapping state indices to state abbreviations.
    naics_codes (list): List of NAICS industry codes.
    industry_code (str): The NAICS code for the industry of interest.

    Returns:
    pd.DataFrame: DataFrame containing state abbreviations and their corresponding predictions.
    """
    # Given our parameters, we can now generate a prediction for each state
    all_inputs = [torch.tensor(np.concatenate([
                    np.array([unemployment_rate]).reshape(-1, 1),
                    state_encoder.transform([[state]]).reshape(-1, 1),
                    industry_encoder.transform([[industry_code]]).reshape(-1, 1)
                ], axis=0), dtype=torch.float32).T for state in state_encoder.categories_[0]]

    # Generate predictions using the model
    predictions = np.array([model.predict(input) for input in all_inputs]).squeeze()

    # Create a dataframe with the predictions and state abbreviations
    df_predictions = pd.DataFrame({
        'State': [state_dict[state] for state in state_encoder.categories_[0]],
        'Prediction': predictions
    })

    return df_predictions

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
