import torch
use_gpu = torch.cuda.is_available()
from librosa.core import load
import os
import json
import numpy as np
from scipy.io.wavfile import write
import torch
use_gpu = torch.cuda.is_available()
import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(sr=16000, n_fft=1024, n_mels=80, fmin=0, fmax=8000)
import params
from model import DiffVC
import sys
sys.path.append('../hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN
sys.path.append('../speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path

class AudioProcessingApp:
    def __init__(self, window):
        self.root = window
        self.root.title("Audio Processing App")
        self.audio_source = None
        self.audio_target = None
        self.btn_select_source = tk.Button(self.root, text="Select Source Audio", command=self.select_source_audio)
        self.btn_select_target = tk.Button(self.root, text="Select Target Audio", command=self.select_target_audio)
        self.btn_process = tk.Button(self.root, text="Process Audio", command=self.process_audio)
        self.btn_extenWav = tk.Button(self.root, text="extenWAV", command=self.extenWAV)
        self.btn_extenVideo = tk.Button(self.root, text="Video New", command=self.extenVideo)

        self.btn_select_source.pack(pady=10)
        self.btn_select_target.pack(pady=10)
        self.btn_process.pack(pady=10)
        self.btn_extenWav.pack(pady=10)
        self.btn_extenVideo.pack(pady=10)

    def extenVideo(self):
        # Sử dụng lớp VideoAudioCombiner
        video_path = "./output/mp4/output_video.mp4"
        audio_path = "./output/wav/output_audio.wav"
        output_path = "./video-new/"
        combiner = VideoAudioCombiner(video_path, audio_path, output_path)
        combiner.combine()

    def extenWAV(self):
        main_window = tk.Tk()
        btn_open_audio_concatenator = tk.Button(main_window, text="ghep WAV",
                                                command=open_audio_concatenator)
        btn_open_audio_concatenator.pack()
        main_window.mainloop()

    def select_source_audio(self):
        self.audio_source = filedialog.askopenfilename(parent=self.root, filetypes=[("Audio Files", "*.wav")])
        print(f"Selected Source Audio: {self.audio_source}")

    def select_target_audio(self):
        self.audio_target = filedialog.askopenfilename(parent=self.root, filetypes=[("Audio Files", "*.wav")])
        print(f"Selected Target Audio: {self.audio_target}")

    def process_audio(self):
        if self.audio_source is not None and self.audio_target is not None:
            def get_mel(wav_path):
                wav, _ = load(wav_path, sr=16000)
                wav = wav[:(wav.shape[0] // 256) * 256]
                wav = np.pad(wav, 384, mode='reflect')
                stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
                stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
                mel_spectrogram = np.matmul(mel_basis, stftm)
                log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
                return log_mel_spectrogram

            def get_embed(wav_path):
                wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
                embed = spk_encoder.embed_utterance(wav_preprocessed)
                return embed

            def noise_median_smoothing(x, w=5):
                y = np.copy(x)
                x = np.pad(x, w, "edge")
                for i in range(y.shape[0]):
                    med = np.median(x[i:i + 2 * w + 1])
                    y[i] = min(x[i + w + 1], med)
                return y

            def mel_spectral_subtraction(mel_synth, mel_source, spectral_floor=0.02, silence_window=5,
                                         smoothing_window=5):
                mel_len = mel_source.shape[-1]
                energy_min = 100000.0
                i_min = 0
                for i in range(mel_len - silence_window):
                    energy_cur = np.sum(np.exp(2.0 * mel_source[:, i:i + silence_window]))
                    if energy_cur < energy_min:
                        i_min = i
                        energy_min = energy_cur
                estimated_noise_energy = np.min(np.exp(2.0 * mel_synth[:, i_min:i_min + silence_window]), axis=-1)
                if smoothing_window is not None:
                    estimated_noise_energy = noise_median_smoothing(estimated_noise_energy, smoothing_window)
                mel_denoised = np.copy(mel_synth)
                for i in range(mel_len):
                    signal_subtract_noise = np.exp(2.0 * mel_synth[:, i]) - estimated_noise_energy
                    estimated_signal_energy = np.maximum(signal_subtract_noise, spectral_floor * estimated_noise_energy)
                    mel_denoised[:, i] = np.log(np.sqrt(estimated_signal_energy))
                return mel_denoised
            # loading voice conversion model
            vc_path = '../checkpts/vc/vc_120.pt'  # path to voice conversion model
            generator = DiffVC(params.n_mels, params.channels, params.filters, params.heads,
                               params.layers, params.kernel, params.dropout, params.window_size,
                               params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim,
                               params.beta_min, params.beta_max)
            if use_gpu:
                generator = generator.cuda()
                generator.load_state_dict(torch.load(vc_path))
            else:
                generator.load_state_dict(torch.load(vc_path, map_location='cpu'))
            generator.eval()
            print(f'Number of parameters: {generator.nparams}')
            # loading HiFi-GAN vocoder
            hfg_path = '../checkpts/vocoder/'  # HiFi-GAN path
            with open(hfg_path + 'config.json') as f:
                h = AttrDict(json.load(f))
            if use_gpu:
                hifigan_universal = HiFiGAN(h).cuda()
                hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator')['generator'])
            else:
                hifigan_universal = HiFiGAN(h)
                hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator', map_location='cpu')['generator'])
            _ = hifigan_universal.eval()
            hifigan_universal.remove_weight_norm()
            # loading speaker encoder
            enc_model_fpath = Path('../checkpts/spk_encoder/encoder.pt')  # speaker encoder path
            if use_gpu:
                spk_encoder.load_model(enc_model_fpath, device="cuda")
            else:
                spk_encoder.load_model(enc_model_fpath, device="cpu")
            src_path = self.audio_source  # path to source utterance
            tgt_path = self.audio_target  # path to reference utterance
            mel_source = torch.from_numpy(get_mel(src_path)).float().unsqueeze(0)
            if use_gpu:
                mel_source = mel_source.cuda()
            mel_source_lengths = torch.LongTensor([mel_source.shape[-1]])
            if use_gpu:
                mel_source_lengths = mel_source_lengths.cuda()
            mel_target = torch.from_numpy(get_mel(tgt_path)).float().unsqueeze(0)
            if use_gpu:
                mel_target = mel_target.cuda()
            mel_target_lengths = torch.LongTensor([mel_target.shape[-1]])
            if use_gpu:
                mel_target_lengths = mel_target_lengths.cuda()
            embed_target = torch.from_numpy(get_embed(tgt_path)).float().unsqueeze(0)
            if use_gpu:
                embed_target = embed_target.cuda()
            # performing voice conversion
            mel_encoded, mel_ = generator.forward(mel_source, mel_source_lengths, mel_target, mel_target_lengths,
                                                  embed_target,
                                                  n_timesteps=100, mode='ml')
            mel_synth_np = mel_.cpu().detach().squeeze().numpy()
            mel_source_np = mel_.cpu().detach().squeeze().numpy()
            mel = torch.from_numpy(
                mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)).float().unsqueeze(0)
            if use_gpu:
                mel = mel.cuda()
            output_dir = './output-convert/'

            with torch.no_grad():
                audio_src = hifigan_universal.forward(mel_source).cpu().squeeze().clamp(-1, 1)
                audio_np_src = audio_src.numpy()
                input_file_name = Path(src_path).stem
                output_file_name = f"{input_file_name}_src.wav"
                output_file_path = os.path.join(output_dir, output_file_name)
                write(output_file_path, 16000, audio_np_src)

            with torch.no_grad():
                audio_tgt = hifigan_universal.forward(mel_target).cpu().squeeze().clamp(-1, 1)
                audio_np_tgt = audio_tgt.numpy()
                input_file_name = Path(tgt_path).stem
                output_file_name = f"{input_file_name}_tgt.wav"
                output_file_path = os.path.join(output_dir, output_file_name)
                write(output_file_path, 16000, audio_np_tgt)

            with torch.no_grad():
                audio_kq = hifigan_universal.forward(mel).cpu().squeeze().clamp(-1, 1)
                audio_np_kq = audio_kq.numpy()
                input_file_name_src = Path(src_path).stem
                input_file_name_tgt = Path(tgt_path).stem
                output_file_name = f"{input_file_name_src}_{input_file_name_tgt}_kq.wav"
                output_file_path = os.path.join(output_dir, output_file_name)
                write(output_file_path, 16000, audio_np_kq)
            print("Processing audio...")
            self.root.destroy()
        else:
            print("Please select both source and target audio files.")


from moviepy.editor import VideoFileClip, AudioFileClip
class VideoAudioCombiner:
    def __init__(self, video_path, audio_path, output_path):
        self.video_path = video_path
        self.audio_path = audio_path
        self.output_path = output_path

    def combine(self):
        # Tải video không có âm thanh và âm thanh
        video_clip = VideoFileClip(self.video_path)
        audio_clip = AudioFileClip(self.audio_path)
        # Ghép video và âm thanh lại với nhau
        video_clip = video_clip.set_audio(audio_clip)
        # Lưu video kết quả
        video_clip.write_videofile(self.output_path, codec="libx264")
        # Đóng tệp audio và video
        audio_clip.close()
        video_clip.close()
        print(f"Ghép thành công và lưu vào {self.output_path}")

import tkinter as tk
from tkinter import filedialog
from pydub import AudioSegment

class AudioConcatenatorApp:
    def __init__(self, window):
        self.root = window
        self.root.title("Audio Concatenator")
        self.audio1_path = ""
        self.audio2_path = ""
        self.btn_select_audio1 = tk.Button(self.root, text="Select Audio 1", command=self.select_audio1)
        self.btn_select_audio2 = tk.Button(self.root, text="Select Audio 2", command=self.select_audio2)
        self.btn_concatenate = tk.Button(self.root, text="Concatenate Audio", command=self.concatenate_audio)
        self.btn_select_audio1.pack()
        self.btn_select_audio2.pack()
        self.btn_concatenate.pack()

    def select_audio1(self):
        self.audio1_path = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])

    def select_audio2(self):
        self.audio2_path = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])

    def concatenate_audio(self):
        if self.audio1_path and self.audio2_path:
            audio1 = AudioSegment.from_wav(self.audio1_path)
            audio2 = AudioSegment.from_wav(self.audio2_path)
            combined_audio = audio1 + audio2
            output_path = "./wav-new/combined_audio.wav"
            combined_audio.export(output_path, format="wav")
            print(f"Concatenated audio saved to {output_path}")
        else:
            print("Please select both audio files.")

def open_audio_concatenator():
    root = tk.Tk()
    app = AudioConcatenatorApp(root)
    root.mainloop()
