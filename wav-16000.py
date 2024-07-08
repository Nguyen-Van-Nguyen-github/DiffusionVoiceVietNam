from pydub import AudioSegment

def convert_audio_to_16000Hz(input_path, output_path):
    audio = AudioSegment.from_wav(input_path)
    # Kiểm tra tần số mẫu của âm thanh
    sample_rate = audio.frame_rate
    if sample_rate != 16000:
        # Chuyển đổi tần số mẫu thành 16,000 Hz
        audio = audio.set_frame_rate(16000)
    # Lưu âm thanh đã chuyển đổi
    audio.export(output_path, format="wav")

# Đường dẫn đến tệp âm thanh ban đầu
input_audio_path = "./data/demo/demo-1/VIVOSSPK25_001.wav"

# Đường dẫn đến tệp âm thanh đã chuyển đổi
output_audio_path = "./data/demo/demo-1/VIVOSSPK25_001_16000.wav"

# Gọi hàm để chuyển đổi âm thanh
convert_audio_to_16000Hz(input_audio_path, output_audio_path)
