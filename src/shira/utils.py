from typing import Union
import librosa, pydub, os, time, glob
import numpy as np 
from functools import wraps


sample_rate = 48000 # sample rate use dto train laion music_CLAP checkpoint
max_duration = 20 # trim all loaded audio to this length for resources

# 'latency' wrapper for reporting time spent in executing a function
def latency(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"latency => {func.__name__}: {end_time - start_time:.4f} seconds")
        return result

    return wrapper


# crawl all the local audio files and retrun a single list
@latency
def audiofile_crawler(root_dir: str, extensions: list =["*.wav", "*.mp3"]) -> list:
    audio_files = []

    for ext in extensions:
        for directory, _, _ in os.walk(root_dir):
            audio_files.append(glob.glob(os.path.join(directory, ext)))

    print(f"found {len(audio_files)} images in {root_dir}")

    return audio_files


def read_audio(audio_file: Union[str, os.PathLike]) -> np.ndarray: #read audio file into numpy array/torch tensor from file path
    if not audio_file.endswith(".wav"): # type: ignore
        audio_file = mp3_to_wav(audio_file) # type: ignore
    waveform, _ = librosa.load(audio_file, sr=sample_rate)
    waveform = trimpad_audio(waveform)

    return waveform


# converting mp3 files to .wav for loading
def mp3_to_wav(file: str) -> str:
    outpath = os.path.basename(file).split(".")[0]
    outpath = f"{outpath}.wav" # full fileame derived from original
    sound = pydub.AudioSegment.from_mp3(file)
    sound.export(outpath)

    return outpath

# trimming audio to a fixed length for all tasks
def trimpad_audio(audio: np.ndarray) -> np.ndarray:
    samples = int(sample_rate * max_duration) # calculate total number of samples

    # cut off excess samples if beyong length, or pad to req. length
    if len(audio) > samples:
        audio = audio[:samples]
    else:
        pad_width = samples - len(audio)
        audio = np.pad(audio, (0, pad_width), mode="reflect")

    return audio

# # displays platable audio widget, for notebooks
# def display_audio(audio: Union[np.ndarray, str], srate: int = 22400):
#     if isinstance(audio, np.ndarray):
#         idp_audio(data=audio, rate=srate)
#     else:
#         idp_audio(filename=audio, rate=srate)