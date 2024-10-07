import librosa, pydub, os
import torch
import numpy as np 

sample_rate = 22400
max_duration = 10


def read_audio(audio_file: str) -> torch.Tensor: #read audio file into torch tensor from file path
    if not audio_file.endswith(".wav"):
        audio_file = mp3_to_wav(audio_file)
    waveform, _ = librosa.load(audio_file, sample_rate)
    waveform = trimpad_audio(waveform)

    return torch.as_tensor(waveform)

# converting mp3 files to .wav for loading
def mp3_to_wav(file: str) -> str:
    outpath = os.path.basename(file).split(".")[0]
    outpath = f"{outpath}.wav" # full fileame derived from original
    sound = pydub.AudioSegment.from_mp3(file)
    sound.export(outpath)

    return outpath

# trimmming audio to a fixed length for all tasks
def trimpad_audio(audio: np.ndarray) -> np.ndarray:
    samples = int(sample_rate * max_duration)

    if len(audio) > samples:
        audio = audio[:samples]

    else:
        pad_width = samples - len(audio)
        audio = np.pad(audio, (0, pad_width), mode="reflect")

    return audio
