import tkinter as tk
from tkinter import filedialog
from moviepy.editor import VideoFileClip
from pyannote.audio import Pipeline
import torch
from pydub import AudioSegment
import os

class VideoPlayer:
    def __init__(self, window):
        self.root = window
        self.root.title("HỆ THỐNG CHUYỂN ĐỔI GIỌNG NÓI TIẾNG VIỆT VỚI MÔ HÌNH KHUẾCH TÁN")
        self.video_source = None
        self.video_clip = None
        self.is_playing = False
        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()
        self.btn_open = tk.Button(self.root, text="Open Video", command=self.open_video)
        self.btn_open.pack(pady=10)
        # self.btn_play = tk.Button(self.root, text="Play", command=self.play)
        # self.btn_play.pack(pady=10)
        # self.btn_pause = tk.Button(self.root, text="Pause", command=self.pause)
        # self.btn_pause.pack(pady=10)
        # self.btn_stop = tk.Button(self.root, text="Stop", command=self.stop)
        # self.btn_stop.pack(pady=10)
        self.btn_sperker = tk.Button(self.root, text="Sperker", command=self.sperker)
        self.btn_sperker.pack(pady=10)

    def sperker(self):
        audio_file = "./output/wav/output_audio.wav"
        output_folder = "./speaker/"
        audio_processor = SpeakerDiarizationAndAudioCut(audio_file, output_folder)
        audio_processor.initialize()
        audio_processor.process()
        self.root.destroy()

    def open_video(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.video_source = file_path
            self.video_clip = VideoFileClip(self.video_source)
            self.play()
            cut_video = CutVideo(self.video_source, "./output/wav/output_audio.wav", "output/mp4/output_video.mp4")
            cut_video.cut_and_save()
            # self.play()
            self.root.destroy()

    def play(self):
        if self.video_clip is not None:
            self.is_playing = True
            self.video_clip.preview()

    def pause(self):
        self.is_playing = False
        self.stop_video()

    def stop(self):
        self.is_playing = False
        self.stop_video()

    def stop_video(self):
        if self.video_clip is not None:
            self.video_clip.reader.close()

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
