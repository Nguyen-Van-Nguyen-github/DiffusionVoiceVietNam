import tkinter as tk
from VideoPlayer import VideoPlayer  # Đảm bảo đã import lớp VideoPlayer
from processAudio import AudioProcessingApp

class MainApp:
    def __init__(self, window):
        self.root = window
        self.root.title("HỆ THỐNG CHUYỂN ĐỔI GIỌNG NÓI TIẾNG VIỆT VỚI MÔ HÌNH KHUẾCH TÁN")

        # Tạo nút để mở VideoPlayer
        self.btn_open_video_player = tk.Button(self.root, text="Tền Xử Lý Dữ Liệu", command=self.open_video_player)
        self.btn_open_video_player.pack()

        self.btn_process_audio = tk.Button(self.root, text="Convert Audio", command=self.process_audio)
        self.btn_process_audio.pack()

    def open_video_player(self):
        # Tạo một thể hiện của lớp VideoPlayer và hiển thị nó
        video_player_window = tk.Toplevel(self.root)
        video_player = VideoPlayer(video_player_window)

    def process_audio(self):
        audio_processing_window = tk.Toplevel(self.root)
        audio_processing = AudioProcessingApp(audio_processing_window)

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
