from pydub import AudioSegment
from tkinter import filedialog
import tkinter as tk

class AudioConcatenator:
    def __init__(self, window):
        self.root = window
        self.root.title("Audio Concatenator")

        self.audio1 = None
        self.audio2 = None

        self.btn_select_audio1 = tk.Button(self.root, text="Select Audio 1", command=self.select_audio1)
        self.btn_select_audio2 = tk.Button(self.root, text="Select Audio 2", command=self.select_audio2)
        self.btn_concatenate = tk.Button(self.root, text="Concatenate Audio", command=self.concatenate_audio)

        self.btn_select_audio1.pack()
        self.btn_select_audio2.pack()
        self.btn_concatenate.pack()

    def select_audio1(self):
        audio_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if audio_path:
            self.audio1 = AudioSegment.from_wav(audio_path)

    def select_audio2(self):
        audio_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if audio_path:
            self.audio2 = AudioSegment.from_wav(audio_path)

    def concatenate_audio(self):
        if self.audio1 and self.audio2:
            combined_audio = self.audio1 + self.audio2
            combined_audio.export("./wav-new/combined_audio.wav", format="wav")
            print("Concatenation completed and saved to combined_audio.wav")
        else:
            print("Please select both audio files.")

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = AudioConcatenator(root)
#     root.mainloop()
