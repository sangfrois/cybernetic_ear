import librosa
import numpy as np
from .base_stream import BaseStream
import torch
import torch.nn as nn

class TimbreAutoencoder(nn.Module):
    """
    A 1D Convolutional Autoencoder for learning a compressed representation of timbral texture.
    """
    def __init__(self, chunk_size=2048):
        super(TimbreAutoencoder, self).__init__()

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Calculate the flattened size dynamically
        self.flattened_size = self._get_flattened_size(chunk_size)
        self.fc1 = nn.Linear(self.flattened_size, 128)

        # --- Decoder ---
        self.decoder_fc = nn.Linear(128, self.flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=2, stride=2),
        )

    def _get_flattened_size(self, chunk_size):
        x = torch.zeros(1, 1, chunk_size)
        x = self.encoder(x)
        return x.numel()

    def encode(self, x):
        """
        Encodes the input chunk into a compressed representation.
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0) # Add batch dimension if missing
        if len(x.shape) == 2:
            x = x.unsqueeze(1) # Add channel dimension if missing
        
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1) # Flatten
        encoded = self.fc1(encoded)
        return encoded

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0) # Add batch dimension if missing
        if len(x.shape) == 2:
            x = x.unsqueeze(1) # Add channel dimension if missing
        
        # Encode
        encoded = self.encode(x)

        # Decode
        decoded = self.decoder_fc(encoded)
        decoded = decoded.view(decoded.size(0), 32, -1) # Unflatten
        decoded = self.decoder(decoded)

        return decoded

import os

class TimbreStream(BaseStream):
    """
    Analyzes the timbral and spectral features of the audio stream.
    """
    def __init__(self, rate=22050, chunk_size=2048):
        self.rate = rate
        self.chunk_size = chunk_size
        self.prev_spectrum = None
        self.cnn_model = TimbreAutoencoder(chunk_size=self.chunk_size)
        
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core', 'timbre_cnn.pth'))
        if os.path.exists(model_path):
            self.cnn_model.load_state_dict(torch.load(model_path))
            self.cnn_model.eval()
            print("Loaded pre-trained TimbreAutoencoder model.")
        else:
            print("Warning: Pre-trained TimbreAutoencoder model not found. Using an untrained model.")

    def process_chunk(self, chunk, feature_bus):
        """
        Processes a chunk to extract timbral features and updates the feature bus.
        """
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        # --- 1D CNN for Timbral Texture ---
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
        timbre_texture = self.cnn_model.encode(chunk_tensor).detach().numpy().flatten()
        feature_bus.update_feature("timbre_texture", timbre_texture)

        # --- Standard MIR Features ---
        centroid = librosa.feature.spectral_centroid(y=chunk, sr=self.rate).mean()
        feature_bus.update_feature("spectral_centroid", centroid)

        flatness = librosa.feature.spectral_flatness(y=chunk).mean()
        feature_bus.update_feature("spectral_flatness", flatness)

        rolloff = librosa.feature.spectral_rolloff(y=chunk, sr=self.rate).mean()
        feature_bus.update_feature("spectral_rolloff", rolloff)

        mfccs = librosa.feature.mfcc(y=chunk, sr=self.rate, n_mfcc=13)
        feature_bus.update_feature("mfccs", mfccs.mean(axis=1))

        # --- Spectral Flux ---
        stft = librosa.stft(chunk)
        current_spectrum = np.abs(stft)

        if self.prev_spectrum is not None and current_spectrum.shape == self.prev_spectrum.shape:
            flux = np.linalg.norm(current_spectrum - self.prev_spectrum)
            feature_bus.update_feature("spectral_flux", flux)

        self.prev_spectrum = current_spectrum

    def process_file(self, audio_buffer: np.ndarray) -> dict:
        """
        Processes an entire audio file to extract timbral features.
        """
        print("Processing Timbre Features...")
        
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_buffer, sr=self.rate)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_buffer)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_buffer, sr=self.rate)[0]
        mfccs = librosa.feature.mfcc(y=audio_buffer, sr=self.rate, n_mfcc=13)
        
        onset_env = librosa.onset.onset_strength(y=audio_buffer, sr=self.rate)
        spectral_flux = np.pad(onset_env, (0, len(spectral_centroid) - len(onset_env)), 'constant')

        timbre_texture = np.zeros_like(spectral_centroid)

        return {
            "spectral_centroid": spectral_centroid,
            "spectral_flatness": spectral_flatness,
            "spectral_rolloff": spectral_rolloff,
            "mfccs": mfccs,
            "spectral_flux": spectral_flux,
            "timbre_texture": timbre_texture
        }

