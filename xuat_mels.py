import os
import numpy as np
import torch
use_gpu = torch.cuda.is_available()
import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(sr=16000, n_fft=1024, n_mels=80, fmin=0, fmax=8000)
import sys
sys.path.append('hifi-gan/')
sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path
# enc_model_fpath = Path('checkpts/spk_encoder/encoder.pt')
# spk_encoder.load_model(enc_model_fpath, device="cpu")

def get_mel(wav_path):
    wav, _ = load(wav_path, sr=16000)
    wav = wav[:(wav.shape[0] // 256)*256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram

# def get_embed(wav_path):
#     wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
#     embed = spk_encoder.embed_utterance(wav_preprocessed)
#     return embed
# Đường dẫn đến thư mục chứa các tệp âm thanh đầu vào
input_folder = './data/wavs/VIVOSSPK46'

# Đường dẫn đến thư mục lưu trữ các tệp .npy đầu ra
output_folder = './data/mels/VIVOSSPK46'

# Tạo thư mục đầu ra nếu nó không tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Duyệt qua tất cả các tệp âm thanh trong thư mục đầu vào
for filename in os.listdir(input_folder):
    if filename.endswith('.wav'):
        wav_path = os.path.join(input_folder, filename)

        # Gọi hàm get_embed để tính toán nhúng
        mels = get_mel(wav_path)

        # Tạo tên tệp .npy tương ứng
        output_filename = os.path.splitext(filename)[0] + '_mel.npy'
        output_path = os.path.join(output_folder, output_filename)

        # Lưu giá trị nhúng vào tệp .npy
        np.save(output_path, mels)
