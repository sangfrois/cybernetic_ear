import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import librosa
import os

from .streams.stream_timbre import TimbreAutoencoder

def train_timbre_autoencoder(audio_path, model_save_path, num_epochs=100, learning_rate=0.001, batch_size=32):
    """
    Trains the TimbreAutoencoder model on an audio file using the GPU.

    Args:
        audio_path (str): The path to the audio file.
        model_save_path (str): The path to save the trained model.
        num_epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.
        batch_size (int): The number of samples per batch.
    """
    # --- 1. Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- 2. Load and Preprocess Data ---
    print(f"Loading audio from: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050)
    
    chunk_size = 2048
    chunks = [y[i:i+chunk_size] for i in range(0, len(y) - chunk_size, chunk_size)]
    
    data = torch.tensor(np.array(chunks), dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- 3. Initialize Model, Loss, and Optimizer ---
    model = TimbreAutoencoder(chunk_size=chunk_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 4. Training Loop ---
    print("Starting training...")
    for epoch in range(num_epochs):
        for batch in dataloader:
            batch_chunks = batch[0].to(device)
            
            # Forward pass
            output = model(batch_chunks)
            loss = criterion(output, batch_chunks)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # --- 5. Save the Trained Model ---
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to: {model_save_path}")

if __name__ == '__main__':
    audio_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'boom-bap-drums-loop-tribe_77bpm_A#_minor.wav'))
    model_save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'core', 'timbre_cnn.pth'))
    train_timbre_autoencoder(audio_file, model_save_path)