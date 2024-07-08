from PyQt5.QtGui import QPixmap
from moviepy.editor import VideoFileClip
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
import json
import numpy as np
from scipy.io.wavfile import write
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
import subprocess
from PyQt5.QtWidgets import QFileDialog, QLabel
import pydub

class VideoPlayerController:
    def __init__(self, ui):
        self.tenKq = None
        self.ui = ui
        self.ui.linkVideo.clicked.connect(self.choose_video)
        self.ui.tachMp4Wav.clicked.connect(self.cut_video)  # Kết nối nút tachMp4Wav với hàm cut_video
        self.ui.speaker.clicked.connect(self.sperker)

        self.ui.srcFile.clicked.connect(self.audio_source)
        self.ui.tgtFile.clicked.connect(self.audio_target)
        self.ui.convertAudio.clicked.connect(self.process_audio)

        self.ui.chonWav1.clicked.connect(self.audio_source)
        self.ui.chonWav2.clicked.connect(self.audio_target)
        self.ui.noiWav.clicked.connect(self.combine_audio)

        self.ui.chonWav.clicked.connect(self.audio_source)
        self.ui.chonMp4.clicked.connect(self.audio_source_mp4)
        self.ui.gopMp4Wav.clicked.connect(self.ghep_mp4_wav)
        #self.ui.phatVideoBtn.clicked.connect(self.play_video)


    # def play_video(self):
    #     subprocess.Popen(["python", "play_video.py"])




    def choose_video(self):
        video_file = QFileDialog.getOpenFileName(None, "Chọn Video", "", "Video Files (*.mp4 *.avi *.mov)")[0]
        if video_file:
            self.ui.label_linkVideo.setText(video_file)  # Cập nhật nhãn để hiển thị đường dẫn video
            self.video_source = video_file

    def cut_video(self):
        if hasattr(self, 'video_source'):
            cut_video = CutVideo(self.video_source, "./data/video-output/wav/output_audio.wav", "./data/video-output/mp4/output_video.mp4")
            cut_video.cut_and_save()

    def sperker(self):
        audio_file = "./data/video-output/wav/output_audio.wav"
        output_folder = "./data/speaker/"
        audio_processor = SpeakerDiarizationAndAudioCut(audio_file, output_folder)
        audio_processor.initialize()
        audio_processor.process()

    def audio_source(self):
        self.audio_source = QFileDialog.getOpenFileName(None, "Chọn Video", "", "Video Files (*.wav)")[0]
        if self.audio_source:
            self.ui.label_linkVideo.setText(self.audio_source)  # Cập nhật nhãn để hiển thị đường dẫn video
            self.audio_source = self.audio_source

    def audio_target(self):
        self.audio_target = QFileDialog.getOpenFileName(None, "Chọn Video", "", "Video Files (*.wav)")[0]
        if self.audio_target:
            self.ui.label_linkVideo.setText(self.audio_target)
            # Cập nhật nhãn để hiển thị đường dẫn video
            self.audio_target = self.audio_target

    def audio_source_mp4(self):
        self.audio_source_mp4 = QFileDialog.getOpenFileName(None, "Chọn Video", "", "Video Files (*.mp4)")[0]
        if self.audio_source_mp4:
            self.ui.label_linkVideo.setText(self.audio_source_mp4)  # Cập nhật nhãn để hiển thị đường dẫn video
            self.audio_source_mp4 = self.audio_source_mp4

    def ghep_mp4_wav(self):
        from moviepy.editor import VideoFileClip, AudioFileClip, clips_array
        video_clip = VideoFileClip(self.audio_source_mp4)
        audio_clip = AudioFileClip(self.audio_source)
        # Ghép video và âm thanh lại với nhau
        video_clip = video_clip.set_audio(audio_clip)
        # Lưu video kết quả
        output_path = "./data/video-convert-full/video_convert.mp4"
        video_clip.write_videofile(output_path, codec="libx264")
        audio_clip.close()
        video_clip.close()
        print(f"Ghép thành công và lưu vào {output_path}")

    def combine_audio(self):
        if self.audio_source and self.audio_target:
            audio1 = pydub.AudioSegment.from_wav(self.audio_source)
            audio2 = pydub.AudioSegment.from_wav(self.audio_target)
            combined_audio = audio1 + audio2
            combined_audio.export("./data/wav-new/wav-new.wav", format="wav")
            print("Audio files combined and saved as 'combined_audio.wav'")
        else:
            print("Please select both source and target audio files.")

    def process_audio(self):
        if self.audio_source is not None and self.audio_target is not None:
            def get_mel(wav_path):
                wav, _ = load(wav_path, sr=22050)
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
            vc_path = './checkpts/vc/vc_120.pt'  # path to voice conversion model
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
            hfg_path = './checkpts/vocoder/'  # HiFi-GAN path
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
            enc_model_fpath = Path('./checkpts/spk_encoder/encoder.pt')  # speaker encoder path
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
                                                  n_timesteps=50, mode='ml')
            mel_synth_np = mel_.cpu().detach().squeeze().numpy()
            mel_source_np = mel_.cpu().detach().squeeze().numpy()
            mel = torch.from_numpy(
                mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)).float().unsqueeze(0)
            if use_gpu:
                mel = mel.cuda()
            output_dir = './data/output-convert/'

            with torch.no_grad():
                audio_src = hifigan_universal.forward(mel_source).cpu().squeeze().clamp(-1, 1)
                audio_np_src = audio_src.numpy()
                input_file_name = Path(src_path).stem
                output_file_name = f"{input_file_name}_src.wav"
                output_file_path = os.path.join(output_dir, output_file_name)
                write(output_file_path, 22050, audio_np_src)

            with torch.no_grad():
                audio_tgt = hifigan_universal.forward(mel_target).cpu().squeeze().clamp(-1, 1)
                audio_np_tgt = audio_tgt.numpy()
                input_file_name = Path(tgt_path).stem
                output_file_name = f"{input_file_name}_tgt.wav"
                output_file_path = os.path.join(output_dir, output_file_name)
                write(output_file_path, 22050, audio_np_tgt)

            with torch.no_grad():
                audio_kq = hifigan_universal.forward(mel).cpu().squeeze().clamp(-1, 1)
                audio_np_kq = audio_kq.numpy()
                input_file_name_src = Path(src_path).stem
                input_file_name_tgt = Path(tgt_path).stem
                output_file_name = f"{input_file_name_src}_{input_file_name_tgt}_kq.wav"
                output_file_path = os.path.join(output_dir, output_file_name)
                write(output_file_path, 22050, audio_np_kq)
            print("Processing audio...")
        else:
            print("Please select both source and target audio files.")

class CutVideo:
    def __init__(self, input_video_path, output_audio_path, output_video_path):
        self.input_video_path = input_video_path
        self.output_audio_path = output_audio_path
        self.output_video_path = output_video_path

    def cut_and_save(self):
        # Load video và trích xuất audio
        video_clip = VideoFileClip(self.input_video_path)
        audio_clip = video_clip.audio
        # Lưu audio vào tệp .wav
        audio_clip.write_audiofile(self.output_audio_path)
        # Lưu video không tiếng nói
        video_clip_without_audio = video_clip.set_audio(None)
        video_clip_without_audio.write_videofile(self.output_video_path, codec="libx264")
        # Đóng tệp audio và video
        audio_clip.close()
        video_clip.close()
        print(f"Chuyển đổi thành công từ {self.input_video_path} thành {self.output_audio_path} và {self.output_video_path}")


class SpeakerDiarizationAndAudioCut:
    def __init__(self, audio_file, output_folder):
        self.audio_file = audio_file
        self.output_folder = output_folder
        self.pipeline = None
        self.audio = None

    def initialize(self):
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token="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")
        self.pipeline.to(torch.device("cpu"))
        self.audio = AudioSegment.from_wav(self.audio_file)

    def process(self):
        if not self.pipeline or not self.audio:
            print("Initialize the pipeline first.")
            return
        diarization = self.pipeline(self.audio_file)
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_time = round(turn.start * 1000, 1)
            end_time = round(turn.end * 1000, 1)
            segment = self.audio[start_time:end_time]
            output_file = f"{speaker}.wav"
            output_file_new = os.path.join(self.output_folder, output_file)
            if os.path.exists(output_file_new):
                i = 0
                while os.path.exists(output_file_new):
                    i += 1
                    output_file = f"{speaker}_{i}.wav"
                    output_file_new = os.path.join(self.output_folder, output_file)
            segment.export(output_file_new, format="wav")
            print(f"Saved: {output_file}")




