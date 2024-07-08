# import os
# import argparse
# import json
# import os
# import numpy as np
# import IPython.display as ipd
# from tqdm import tqdm
# from scipy.io.wavfile import write
#
# import torch
# use_gpu = torch.cuda.is_available()
#
# import librosa
# from librosa.core import load
# from librosa.filters import mel as librosa_mel_fn
# mel_basis = librosa_mel_fn(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)
#
# import params
# from model import DiffVC
#
# import sys
# sys.path.append('hifi-gan/')
# from env import AttrDict
# from models import Generator as HiFiGAN
#
# sys.path.append('speaker_encoder/')
# from encoder import inference as spk_encoder
# from pathlib import Path
#
# enc_model_fpath = Path('checkpts/spk_encoder/encoder_91.pt')
# spk_encoder.load_model(enc_model_fpath, device="cpu")
#
#
# def get_embed(wav_path):
#     wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
#     embed = spk_encoder.embed_utterance(wav_preprocessed)
#     return embed
# # Đường dẫn đến thư mục chứa các tệp âm thanh đầu vào
# input_folder = './data/wavs/VIVOSSPK46'
#
# # Đường dẫn đến thư mục lưu trữ các tệp .npy đầu ra
# output_folder = './data/embeds/VIVOSSPK46'
#
# # Tạo thư mục đầu ra nếu nó không tồn tại
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # Duyệt qua tất cả các tệp âm thanh trong thư mục đầu vào
# for filename in os.listdir(input_folder):
#     if filename.endswith('.wav'):
#         wav_path = os.path.join(input_folder, filename)
#
#         # Gọi hàm get_embed để tính toán nhúng
#         embed = get_embed(wav_path)
#
#         # Tạo tên tệp .npy tương ứng
#         output_filename = os.path.splitext(filename)[0] + '_embed.npy'
#         output_path = os.path.join(output_folder, output_filename)
#
#         # Lưu giá trị nhúng vào tệp .npy
#         np.save(output_path, embed)
#
#
# # import os
# #
# # # Thư mục chứa các tệp
# # folder_path = './data/embeds/VIVOSSPK46'
# #
# # # Lặp qua tất cả các tệp trong thư mục
# # for filename in os.listdir(folder_path):
# #     if filename.endswith('.npy'):
# #         # Tạo tên mới cho tệp
# #         new_filename = filename.replace('.npy', '_embed.npy')
# #
# #         # Thay đổi tên tệp
# #         os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
# #         print(f'Tên tệp {filename} đã được thay đổi thành {new_filename}')




import os

import numpy as np
import torch
use_gpu = torch.cuda.is_available()

import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)

import params
from model import DiffVC

import sys
sys.path.append('hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path

enc_model_fpath = Path('checkpts/spk_encoder/encoder-80-nmels.pt')
spk_encoder.load_model(enc_model_fpath, device="cpu")

def get_embed(wav_path):
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed

# Đường dẫn đến thư mục chứa các thư mục con với các tệp âm thanh đầu vào
input_root_folder = './A/wavs/'
output_root_folder = './A/embeds/'

# Tạo thư mục đầu ra nếu nó không tồn tại
if not os.path.exists(output_root_folder):
    os.makedirs(output_root_folder)

# Duyệt qua tất cả các thư mục con và tệp âm thanh trong thư mục đầu vào
for dirpath, dirnames, filenames in os.walk(input_root_folder):
    for filename in filenames:
        if filename.endswith('.wav'):
            wav_path = os.path.join(dirpath, filename)

            # Gọi hàm get_embed để tính toán nhúng
            embed = get_embed(wav_path)

            # Tạo đường dẫn đầu ra tương ứng trong thư mục nhúng
            relative_path = os.path.relpath(wav_path, input_root_folder)
            output_dir = os.path.join(output_root_folder, os.path.dirname(relative_path))
            output_filename = os.path.splitext(filename)[0] + '_embed.npy'
            output_path = os.path.join(output_dir, output_filename)

            # Tạo thư mục đầu ra nếu nó không tồn tại
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Lưu giá trị nhúng vào tệp .npy
            np.save(output_path, embed)
