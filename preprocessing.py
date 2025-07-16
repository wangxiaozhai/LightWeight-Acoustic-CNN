import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

input_folder_path = 'data'
output_folder_path = 'features'

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

for file_name in os.listdir(input_folder_path):
    if file_name.endswith('.wav'):
        input_file_path = os.path.join(input_folder_path, file_name)

        y, sr = librosa.load(input_file_path, sr=None)
        y = y / np.max(np.abs(y))

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=90, fmax=sr//2)
        S_dB = librosa.power_to_db(S, ref=np.max)

        np.save(os.path.join(output_folder_path, file_name.replace('.wav', '.npy')), S_dB)

        plt.figure(figsize=(6, 4))
        librosa.display.specshow(S_dB, y_axis='mel', x_axis='time', sr=sr, fmax=sr/2)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram (Simplified)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder_path, file_name.replace('.wav', '.png')))
        plt.close()

        print(f"Processed: {file_name}")
