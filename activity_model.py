import torch
import numpy as np
import os
from pathlib import Path
from AAECG import Encoder, Decoder

class ActivityRecognizer:
    def __init__(self):
        print("Loading AnoBeat ECG anomaly detection model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Model paths
        model_path = Path(os.path.dirname(__file__)) / "AnoBeat"
        encoder_path = model_path / "encoder.mod"
        decoder_path = model_path / "decoder.mod"
        threshold_path = model_path / "thr.npy"
        
        # Load models
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
        
        self.encoder.eval()
        self.decoder.eval()
        
        # Load threshold for anomaly detection
        self.threshold = float(np.load(threshold_path))
        
        print("ECG anomaly detection model loaded successfully")
        
        # ECG buffer for storing signal
        self.ecg_buffer = []
        self.signal_length = 256  # Expected input length
        
    def add_ecg_sample(self, sample):
        """Add a single ECG sample to buffer"""
        self.ecg_buffer.append(sample)
        
        # Keep buffer at the right size
        if len(self.ecg_buffer) > self.signal_length:
            self.ecg_buffer = self.ecg_buffer[-self.signal_length:]
    
    def predict(self, frames=None):
        """Process ECG buffer and predict cardiac activity"""
        # For compatibility with the original interface that expected frames
        # This version uses the internally stored ECG buffer instead
        
        if len(self.ecg_buffer) < self.signal_length:
            return f"collecting ECG data... ({len(self.ecg_buffer)}/{self.signal_length})"
        
        # Prepare ECG data
        ecg_input = np.array(self.ecg_buffer[-self.signal_length:])
        ecg_tensor = torch.tensor(ecg_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Run through model
        with torch.no_grad():
            latent = self.encoder(ecg_tensor)
            reconstruction = self.decoder(latent)
        
        # Calculate reconstruction error
        error = torch.mean((reconstruction - ecg_tensor) ** 2).item()
        
        # Determine anomaly status
        if error > self.threshold:
            return "arrhythmia"
        else:
            return "normal_rhythm"
    
    def get_current_data(self):
        """Get current ECG data and its reconstruction for visualization"""
        if len(self.ecg_buffer) < self.signal_length:
            return None, None, 0
            
        # Prepare ECG data
        ecg_input = np.array(self.ecg_buffer[-self.signal_length:])
        ecg_tensor = torch.tensor(ecg_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Run through model
        with torch.no_grad():
            latent = self.encoder(ecg_tensor)
            reconstruction = self.decoder(latent)
        
        # Calculate reconstruction error
        error = torch.mean((reconstruction - ecg_tensor) ** 2).item()
        
        return ecg_input, reconstruction.squeeze().cpu().numpy(), error
