import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from MyAnoBeat import MyAnoBeat

def load_ecg_file(file_path):
    """Load ECG data from file"""
    print(f"Loading ECG data from {file_path}")
    if file_path.endswith('.npy'):
        data = np.load(file_path)
    elif file_path.endswith('.csv'):
        data = np.loadtxt(file_path, delimiter=',')
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Ensure proper shape and length
    if len(data) > 280:
        # Truncate to expected length
        data = data[:280]
    elif len(data) < 280:
        # Pad with zeros
        data = np.pad(data, (0, 280 - len(data)))
    
    # Convert to tensor with correct shape [1, 1, 280]
    tensor = torch.from_numpy(data).float().view(1, 1, 280)
    return tensor

# Directory settings
data_dir = "new_ecg_data"  # Directory with ECG files
results_dir = "ecg_analysis_results"  # Directory to save results

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Initialize the model
print("Loading AnoBeat model...")
anobeat = MyAnoBeat(model_path="models/AnoBeat/", threshold=0.5)
print("Model loaded successfully")

# Process each ECG file in the directory
ecg_files = [f for f in os.listdir(data_dir) if f.endswith(('.npy', '.csv'))]
print(f"Found {len(ecg_files)} ECG files to analyze")

# Create summary table for results
print("\n" + "="*60)
print(f"{'Filename':30} | {'Anomaly Score':15} | {'Prediction':10}")
print("="*60)

for file_name in ecg_files:
    file_path = os.path.join(data_dir, file_name)
    
    try:
        # Load the ECG data
        ecg_data = load_ecg_file(file_path)
        
        # Analyze using the model
        analysis = anobeat.analyze_single_ecg(ecg_data)
        
        # Get results
        score = analysis['anomaly_score']
        is_abnormal = analysis['is_abnormal']
        prediction = "ABNORMAL" if is_abnormal else "NORMAL"
        
        # Print results
        print(f"{file_name:30} | {score:.6f}        | {prediction:10}")
        
        # Create visualization
        fig = anobeat.visualize_anomalies(ecg_data, 
                                        title=f"ECG Analysis: {file_name}")
        
        # Save visualization to results directory
        output_path = os.path.join(results_dir, f"{os.path.splitext(file_name)[0]}_analysis.png")
        fig.savefig(output_path)
        plt.close(fig)
        
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

print("="*60)
print(f"\nAnalysis complete. Results saved to {results_dir}")

# Create a comparison figure showing all analyzed ECGs and their scores
try:
    plt.figure(figsize=(15, 10))
    
    scores = []
    file_names = []
    
    for i, file_name in enumerate(ecg_files):
        file_path = os.path.join(data_dir, file_name)
        ecg_data = load_ecg_file(file_path)
        
        # Get the raw signal data for plotting
        signal = ecg_data.squeeze().numpy()
        
        # Get anomaly score
        analysis = anobeat.analyze_single_ecg(ecg_data)
        score = analysis['anomaly_score']
        is_abnormal = analysis['is_abnormal']
        
        scores.append(score)
        file_names.append(os.path.splitext(file_name)[0])
        
        # Add subplot for this ECG
        plt.subplot(len(ecg_files), 1, i+1)
        plt.plot(signal)
        plt.title(f"{file_name} - Score: {score:.6f} ({'Abnormal' if is_abnormal else 'Normal'})")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "all_ecgs_comparison.png"))
    plt.close()
    
    # Create a bar chart of anomaly scores
    plt.figure(figsize=(10, 6))
    bars = plt.bar(file_names, scores)
    
    # Color the bars based on prediction
    for i, score in enumerate(scores):
        if score >= anobeat.threshold:
            bars[i].set_color('red')
        else:
            bars[i].set_color('green')
    
    plt.axhline(y=anobeat.threshold, color='black', linestyle='--', label=f'Threshold ({anobeat.threshold})')
    plt.xlabel('ECG File')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Scores for ECG Files')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(results_dir, "anomaly_scores_comparison.png"))
    
    print(f"Created summary visualizations in {results_dir}")
    
except Exception as e:
    print(f"Error creating summary visualization: {e}")