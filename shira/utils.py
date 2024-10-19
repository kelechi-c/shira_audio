from typing import Tuple, Union, List
import librosa, os, time
import numpy as np
import soundfile as sf
from functools import wraps
from pathlib import Path
from tqdm import tqdm

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
def audiofile_crawler(directory: Union[str, Path] = Path.home()) -> Tuple[List[str], int]:
    audio_extensions = (".mp3", ".wav", ".flac", ".ogg")
    # audio_files = []

    dir_path = Path(directory)
    print(dir_path)

    # Use a generator expression to find all files with the given extensions
    audio_files = [str(file) for ext in tqdm(audio_extensions, ncols=50) for file in dir_path.rglob(f"*{ext}")]

    matching_files = filter_files(audio_files)
    
    print(f"{len(matching_files)} valid files out of {len(audio_files)} files.")

    return matching_files, len(matching_files)


def read_audio(audio_file: Union[str, os.PathLike]) -> np.ndarray: #read audio file into numpy array/torch tensor from file path
    waveform, _ = librosa.load(audio_file, sr=sample_rate)
    waveform = trimpad_audio(waveform)

    return waveform

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

# filter corrupt files and return list of soundfile-readable files only
def filter_files(audio_files: List) -> List[str]:
    sane_files = []
    
    for file in audio_files:
        try:
            sf.read(file)
            sane_files.append(file)
        except:
            continue
        
    return sane_files