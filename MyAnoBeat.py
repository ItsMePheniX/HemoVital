import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, fbeta_score
import Baselines as b

class MyAnoBeat:
    """
    A wrapper class for AnoBeat model to detect anomalies in ECG signals.
    """
    def __init__(self, model_path="models/AnoBeat/", device=None, threshold=None):
        """
        Initialize the AnoBeat model.
        
        Args:
            model_path: Path to the pre-trained model
            device: Device to run the model on (cuda or cpu)
            threshold: Threshold for anomaly detection (None = auto-calibrate)
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Initialize the model
        self.model = b.AnoBeat(device=self.device, N=280, L=1, nz=50, nef=32)
        
        # Load pre-trained weights
        try:
            self.model.load_model(model_path)
            print(f"AnoBeat model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Warning when loading model: {e}")
        
        self.threshold = threshold
        if self.threshold is None:
            self.threshold = 0.5  # Default threshold
            print(f"Using default threshold: {self.threshold}")
        else:
            print(f"Using custom anomaly threshold: {self.threshold}")
        
    def calibrate_threshold(self, dataloader, target_specificity=0.95):
        """
        Auto-calibrate threshold based on normal ECG data to achieve target specificity.
        
        Args:
            dataloader: DataLoader with validation data
            target_specificity: Target specificity (1-FPR) for threshold selection
            
        Returns:
            Optimal threshold
        """
        print(f"Calibrating threshold to achieve {target_specificity:.2f} specificity...")
        all_scores = []
        all_labels = []
        
        # Process each batch
        for data in dataloader:
            signals = data[0].to(self.device)
            metadata = data[2]
            
            # Debug information
            if 'label' not in metadata:
                print("ERROR: 'label' not found in metadata!")
                continue
                
            # Get binary labels (0 = normal, 1 = abnormal) with explicit conversion
            binary_labels = []
            for label in metadata['label']:
                if label == 'abnormal':
                    binary_labels.append(1)
                else:
                    binary_labels.append(0)
            
            # Get anomaly scores
            try:
                scores = self.model.get_anomaly_score(signals)
                all_scores.extend(scores)
                all_labels.extend(binary_labels)
            except Exception as e:
                print(f"Error calculating anomaly scores: {e}")
                continue
        
        # Validate we have data
        if len(all_scores) == 0 or len(all_labels) == 0:
            print("No valid data for threshold calibration. Using default threshold.")
            return self.threshold
            
        # Check for class distribution
        abnormal_count = sum(all_labels)
        normal_count = len(all_labels) - abnormal_count
        print(f"Class distribution: {normal_count} normal, {abnormal_count} abnormal samples")
        
        if abnormal_count == 0 or normal_count == 0:
            print("WARNING: Only one class present in calibration data. Using default threshold.")
            return self.threshold
        
        # Convert to numpy arrays
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # Calculate ROC curve
        try:
            fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)
            
            # Find threshold that gives target specificity
            target_fpr = 1 - target_specificity
            optimal_idx = np.argmin(np.abs(fpr - target_fpr))
            optimal_threshold = thresholds[optimal_idx]
            
            print(f"Calibrated threshold: {optimal_threshold:.4f}")
            self.threshold = optimal_threshold
            return optimal_threshold
            
        except Exception as e:
            print(f"Error in ROC curve calculation: {e}")
            print("Using default threshold.")
            return self.threshold
        
    def detect_anomalies(self, signal_batch):
        """
        Detect anomalies in a batch of ECG signals.
        
        Args:
            signal_batch: Batch of ECG signals [batch_size, 1, 280]
            
        Returns:
            Dictionary with anomaly scores, predictions and detailed analysis
        """
        # Ensure proper shape
        if len(signal_batch.shape) == 2:
            signal_batch = signal_batch.unsqueeze(1)  # Add channel dimension [B, 1, 280]
        
        signal_batch = signal_batch.to(self.device)
        batch_size = signal_batch.size(0)
        
        # Get latent representations and reconstructions
        with torch.no_grad():
            z = self.model.netE(signal_batch)
            reconstructions = self.model.netD(z)
        
        # Calculate anomaly scores
        anomaly_scores = self.model.get_anomaly_score(signal_batch)
        
        # Calculate pointwise reconstruction errors for each sample
        pointwise_errors = []
        for i in range(batch_size):
            orig = signal_batch[i].cpu().numpy().flatten()
            recon = reconstructions[i].cpu().numpy().flatten()
            error = np.abs(orig - recon)
            pointwise_errors.append(error)
        
        # Make predictions
        predictions = anomaly_scores >= self.threshold
        
        # Create detailed analysis
        results = {
            'anomaly_scores': anomaly_scores,
            'predictions': predictions,
            'reconstructions': reconstructions.cpu().detach(),
            'pointwise_errors': pointwise_errors,
            'is_abnormal': predictions,
            'confidence': np.abs(anomaly_scores - self.threshold) / self.threshold
        }
        
        return results
    
    def analyze_single_ecg(self, signal):
        """
        Analyze a single ECG signal for anomalies.
        
        Args:
            signal: ECG signal tensor
            
        Returns:
            Dictionary with detailed analysis
        """
        # Ensure proper shape
        if len(signal.shape) == 1:
            signal = signal.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
        elif len(signal.shape) == 2:
            signal = signal.unsqueeze(0)  # [1, 1, N]
            
        # Process through the model
        results = self.detect_anomalies(signal)
        
        # Extract single-item results
        analysis = {
            'anomaly_score': results['anomaly_scores'][0],
            'is_abnormal': bool(results['predictions'][0]),
            'confidence': results['confidence'][0],
            'reconstruction': results['reconstructions'][0],
            'pointwise_error': results['pointwise_errors'][0]
        }
        
        return analysis
    
    def visualize_anomalies(self, signal, title=None):
        """
        Visualize anomalies in an ECG signal.
        
        Args:
            signal: ECG signal tensor
            title: Optional title for the plot
        """
        # Get full analysis
        analysis = self.analyze_single_ecg(signal)
        
        # Convert tensors to numpy for plotting
        orig_signal = signal.squeeze().cpu().numpy()
        recon_signal = analysis['reconstruction'].squeeze().cpu().numpy()
        pointwise_error = analysis['pointwise_error']
        
        # Calculate error threshold for highlighting (95th percentile)
        error_threshold = np.percentile(pointwise_error, 95)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot original and reconstructed signals
        axes[0].plot(orig_signal, 'b-', label='Original ECG')
        axes[0].plot(recon_signal, 'r-', label='Reconstructed', alpha=0.7)
        axes[0].set_title("Original vs Reconstructed ECG" if title is None else title)
        axes[0].legend()
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)
        
        # Plot pointwise reconstruction error
        axes[1].plot(pointwise_error, 'g-')
        axes[1].axhline(y=error_threshold, color='r', linestyle='--', 
                        label=f'Error threshold (95th percentile)')
        axes[1].set_title("Pointwise Reconstruction Error")
        axes[1].set_ylabel("Error Magnitude")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot original signal with anomaly highlighting
        axes[2].plot(orig_signal, 'b-', label='Original ECG')
        
        # Highlight anomalous regions
        anomaly_regions = pointwise_error >= error_threshold
        
        # Highlight anomalous points
        anomalous_indices = np.where(anomaly_regions)[0]
        if len(anomalous_indices) > 0:
            axes[2].scatter(anomalous_indices, orig_signal[anomalous_indices], 
                           color='red', s=50, zorder=5, label='Potential anomalies')
        
        axes[2].set_title(f"Detected Anomalies (Score: {analysis['anomaly_score']:.4f}, " + 
                         f"{'Abnormal' if analysis['is_abnormal'] else 'Normal'})")
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Amplitude")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Add overall info as text
        status = "ABNORMAL" if analysis['is_abnormal'] else "NORMAL"
        confidence = analysis['confidence'] * 100
        fig.suptitle(f"ECG Anomaly Analysis - {status} (Confidence: {confidence:.1f}%)", 
                    fontsize=16, y=0.99)
        
        plt.tight_layout()
        plt.show()
        
        return fig
        
    def evaluate(self, dataloader):
        """
        Evaluate the model on a test dataset.
        
        Args:
            dataloader: DataLoader with test data
            
        Returns:
            results: Dictionary with evaluation metrics
        """
        print("Evaluating ECG anomaly detection performance...")
        all_scores = []
        all_labels = []
        
        for data in dataloader:
            signals = data[0].to(self.device)
            metadata = data[2]
            
            # Get binary labels (0 = normal, 1 = abnormal)
            binary_labels = []
            for label in metadata['label']:
                if label == 'abnormal':
                    binary_labels.append(1)
                else:
                    binary_labels.append(0)
            
            # Get anomaly scores
            scores = self.model.get_anomaly_score(signals)
            
            # Add to collected data
            all_scores.extend(scores)
            all_labels.extend(binary_labels)
        
        # Convert to numpy arrays
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels, dtype=int)
        
        # Debug print to check data
        print(f"Processed {len(all_scores)} ECGs: {sum(all_labels)} abnormal, {len(all_labels) - sum(all_labels)} normal")
        
        if len(all_labels) == 0 or len(np.unique(all_labels)) < 2:
            print("WARNING: Not enough data or classes for proper evaluation")
            return {
                'roc_auc': 0.5, 'pr_auc': 0.5, 'f1': 0,
                'f2': 0, 'sensitivity': 0, 'specificity': 0
            }
        
        # Calculate ROC curve and AUC
        try:
            fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
            roc_auc = auc(fpr, tpr)
            
            # Calculate PR curve and AUC
            precision, recall, _ = precision_recall_curve(all_labels, all_scores, pos_label=1)
            pr_auc = auc(recall, precision)
            
            # Calculate F-scores using the threshold
            predictions = all_scores >= self.threshold
            f1 = f1_score(all_labels, predictions)
            f2 = fbeta_score(all_labels, predictions, beta=2)
            
            # Calculate counts for confusion matrix
            tp = np.sum((all_labels == 1) & (predictions == 1))
            fp = np.sum((all_labels == 0) & (predictions == 1))
            tn = np.sum((all_labels == 0) & (predictions == 0))
            fn = np.sum((all_labels == 1) & (predictions == 0))
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            results = {
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'f1': f1,
                'f2': f2,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'scores': all_scores,
                'labels': all_labels
            }
            
            return results
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return {
                'roc_auc': 0.5, 'pr_auc': 0.5, 'f1': 0, 
                'f2': 0, 'sensitivity': 0, 'specificity': 0
            }