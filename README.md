# DiffusionVoiceVietNam
[en] This project is designed with the goal of converting a source voice into a target voice without any loss of the speaker's words or meaning.

[vi] Dự án này được xây dựng nhầm mục  đích chuyển đổi gióng nói (âm thanh) từ giọng nói nguồn thành giọng nói đích mà không làm mất mát từ ngữ của người nói.

## Description
[en] This is a framework that utilizes models such as diffusion voice conversion, HiFi-GAN, real-time voice cloning, and speaker diarization. It is capable of converting the voices of one or multiple speakers. The specific structure follows a process where the input (video) is first split into audio and video, then the audio is separated by speaker (in cases where multiple speakers are present). After that, the source audio is converted into the target voice, and finally, the audio is reconnected with the video to produce the final result.

[vi] Đây là một framework sử dụng các mô hình như diffusion voice conversion, hifi-gan, real-time-voice-clong, speaker diarization. Nó có thể chuyển đổi âm thanh của một hay nhiều người nói khác nhau. 
cấu trúc cụ thể: input(video) -> tách(âm thanh + video) -> tách âm thanh của từng người(đối với đoạn âm thanh có nhiều người) -> chuyển đổi âm thanh nguồn thành âm thanh đích -> kết nối lại âm thanh với video(kết quả).

## DEMO
DEMO. [link](https://www.youtube.com/watch?v=bgewq_irHfU).

## Programming Language
Python 3.8.0. [link](https://www.python.org/downloads/release/python-380/)

## Model of diffusion voice conversion
Document: [link1](https://openreview.net/pdf?id=8c50f-DoWAu), [link2](https://www.isca-archive.org/interspeech_2023/choi23d_interspeech.pdf)

##hifi-gan Model
Document: [link](https://paperswithcode.com/method/hifi-gan). 

Model. [link](https://github.com/jik876/hifi-gan)

## Model of real-time-voice-cloning
Document: [link](https://www.semanticscholar.org/paper/REAL-TIME-VOICE-CLONING-Daspute-Pandit/e3e85e846a07d8e9152ecf6f80238e547707ef1f). 

Model: [link](https://github.com/CorentinJ/Real-Time-Voice-Cloning)

## Model of speaker-diarization
Document: [link1](https://paperswithcode.com/task/speaker-diarization), [link2](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html)

## Trained Models
checkpts/spk_encoder/encoder.pt [link](https://drive.google.com/file/d/1FTr5FXr5rgRF0C5LNyc9xNII4LtMMv4B/view?usp=drive_link)

checkpts/vc/vc_120.pt [link](https://drive.google.com/file/d/1ZWXmKtrtbUebMQAXemkdhVLM3_CpXAFq/view?usp=drive_link)

checkpts/vocoder/config.json [link](https://drive.google.com/file/d/1CXQUV36Flp3RIHDzz62HfXYtHoXJH6h3/view?usp=drive_link)

checkpts/vocoder/generator [link](https://drive.google.com/file/d/1BqYEKJ7b6sbEqKJytkX9eJVRN2OiT2j-/view?usp=drive_link)

## Run application
pthon gui.py







