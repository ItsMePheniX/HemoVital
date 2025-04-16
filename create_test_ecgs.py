import numpy as np
import matplotlib.pyplot as plt
import os

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

test_dir = "new_ecg_data"
create_directory(test_dir)

def generate_normal_ecg(num_samples=280):
    t = np.linspace(0, 2, num_samples)
    
    heart_rate = 60
    base_freq = heart_rate / 60
    
    p_wave = 0.15 * np.sin(2 * np.pi * base_freq * t)
    qrs_complex = np.zeros_like(t)
    t_wave = 0.3 * np.sin(2 * np.pi * base_freq * t)
    
    beat_interval = int(num_samples / (base_freq * 2))
    for i in range(3):
        center = i * beat_interval + int(0.2 * beat_interval)
        if center < num_samples:
            if center-3 >= 0:
                qrs_complex[center-3:center] = -0.2
            
            if center+5 < num_samples:
                qrs_complex[center:center+5] = 1.0
            
            if center+5 < num_samples and center+10 < num_samples:
                qrs_complex[center+5:center+10] = -0.4
    
    p_wave_shifted = np.zeros_like(p_wave)
    t_wave_shifted = np.zeros_like(t_wave)
    
    for i in range(3):
        center = i * beat_interval + int(0.2 * beat_interval)
        if center < num_samples:
            p_center = center - 20
            if p_center > 0:
                p_width = 15
                for j in range(p_width):
                    if p_center-p_width+j >= 0 and p_center-p_width+j < num_samples:
                        p_wave_shifted[p_center-p_width+j] += 0.2 * np.sin(np.pi * j / p_width)
            
            t_center = center + 30
            if t_center < num_samples:
                t_width = 20
                for j in range(t_width):
                    if t_center+j < num_samples:
                        t_wave_shifted[t_center+j] += 0.25 * np.sin(np.pi * j / t_width)
    
    ecg = p_wave_shifted + qrs_complex + t_wave_shifted
    
    baseline_wander = 0.05 * np.sin(2 * np.pi * 0.05 * t)
    noise = 0.03 * np.random.randn(num_samples)
    
    ecg = ecg + baseline_wander + noise
    
    ecg = ecg / np.max(np.abs(ecg))
    
    return ecg

def generate_abnormal_ecg(num_samples=280):
    ecg = generate_normal_ecg(num_samples)
    
    t = np.linspace(0, 2, num_samples)
    
    beat_interval = int(num_samples / 3)
    for i in range(3):
        center = i * beat_interval + int(0.2 * beat_interval)
        if center < num_samples:
            st_start = center + 10
            st_end = center + 30
            if st_start < num_samples and st_end < num_samples:
                ecg[st_start:st_end] += 0.3
    
    center = beat_interval + int(0.2 * beat_interval)
    if center + 15 < num_samples:
        ecg[center:center+15] = 0.8 * np.sin(np.pi * np.arange(15) / 15)
    
    if 3*beat_interval//2 + 30 < num_samples:
        ecg[3*beat_interval//2 - 10:3*beat_interval//2 + 30] = ecg[3*beat_interval//2 - 10:3*beat_interval//2 + 30] * 0.3
    
    pvc_center = 2 * beat_interval - 15
    if pvc_center > 0 and pvc_center + 20 < num_samples:
        ecg[pvc_center-10:pvc_center+20] = 0
        ecg[pvc_center:pvc_center+10] = -1.5
        ecg[pvc_center+10:pvc_center+20] = 1.2
    
    ecg = ecg / np.max(np.abs(ecg))
    
    return ecg

normal_ecg = generate_normal_ecg()
abnormal_ecg = generate_abnormal_ecg()

np.save(os.path.join(test_dir, "normal_ecg.npy"), normal_ecg)
np.save(os.path.join(test_dir, "abnormal_ecg.npy"), abnormal_ecg)

np.savetxt(os.path.join(test_dir, "normal_ecg.csv"), normal_ecg, delimiter=',')
np.savetxt(os.path.join(test_dir, "abnormal_ecg.csv"), abnormal_ecg, delimiter=',')

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(normal_ecg)
plt.title("Normal ECG")
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(abnormal_ecg)
plt.title("Abnormal ECG")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(test_dir, "ecg_comparison.png"))
print(f"ECG test files created in {test_dir}")

print("\nCreated files:")
print(f"1. Normal ECG: {os.path.join(test_dir, 'normal_ecg.npy')} and {os.path.join(test_dir, 'normal_ecg.csv')}")
print(f"2. Abnormal ECG: {os.path.join(test_dir, 'abnormal_ecg.npy')} and {os.path.join(test_dir, 'abnormal_ecg.csv')}")
print(f"3. Visual comparison: {os.path.join(test_dir, 'ecg_comparison.png')}")

print(f"\nNormal ECG - min: {normal_ecg.min():.2f}, max: {normal_ecg.max():.2f}")
print(f"Abnormal ECG - min: {abnormal_ecg.min():.2f}, max: {abnormal_ecg.max():.2f}")