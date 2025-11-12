import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

from .core.networks import GrooveGRU

def generate_synthetic_groove_data(num_samples=1000, sequence_length=20):
    """
    Generates synthetic microtiming deviation data for training the GrooveGRU.
    """
    data = []
    labels = []
    for _ in range(num_samples):
        # Generate a sequence of random microtiming deviations
        sequence = np.random.randn(sequence_length, 1) * 0.02 # Small deviations
        
        # The label is the next deviation in the sequence
        label = np.mean(sequence) * 1.1 # A simple transformation for the label
        
        data.append(sequence)
        labels.append(label)
        
    return torch.tensor(np.array(data), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(1)

def train_groove_gru(model_save_path, num_epochs=50, learning_rate=0.001, batch_size=32):
    """
    Trains the GrooveGRU model on synthetic data.
    """
    # --- 1. Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- 2. Generate Synthetic Data ---
    data, labels = generate_synthetic_groove_data()
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- 3. Initialize Model, Loss, and Optimizer ---
    model = GrooveGRU().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 4. Training Loop ---
    print("Starting GrooveGRU training...")
    try:
        for epoch in range(num_epochs):
            for sequences, labels in dataloader:
                sequences, labels = sequences.to(device), labels.to(device)
                
                # Forward pass
                output = model(sequences)
                loss = criterion(output, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return

    # --- 5. Save the Trained Model ---
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained GrooveGRU model saved to: {model_save_path}")

if __name__ == '__main__':
    model_save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'core', 'groove_gru.pth'))
    train_groove_gru(model_save_path)
