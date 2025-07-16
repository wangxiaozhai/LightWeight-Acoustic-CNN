import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def wav_to_mel(audio_file, output_folder, sr=100, n_mels=64):
    os.makedirs(output_folder, exist_ok=True)
    y, sr = librosa.load(audio_file, sr=sr)
    file_base = os.path.splitext(os.path.basename(audio_file))[0]
    mel_spect = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=256,
        hop_length=16,
        n_mels=n_mels,
        fmax=sr / 2
    )
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    npy_path = os.path.join(output_folder, f"{file_base}.npy")
    np.save(npy_path, mel_spect)

    plt.figure(figsize=(6, 4))
    librosa.display.specshow(mel_spect, sr=sr, x_axis='time', y_axis='mel', fmax=sr / 2)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram (Simplified)')
    plt.tight_layout()

    png_path = os.path.join(output_folder, f"{file_base}.png")
    plt.savefig(png_path)
    plt.close()

    print(f"[INFO] Saved: {png_path}")
    print(f"[INFO] Saved: {npy_path}")


if __name__ == "__main__":
    # Example
    audio_file = "data/"
    output_folder = "features"
    wav_to_mel(audio_file, output_folder)
