
waveform, sample_rate = torchaudio.load("./A/NguyenVanNguyen/C2.wav", normalize=True)
transform = transforms.MelSpectrogram(sample_rate)
mel_specgram = transform(waveform)