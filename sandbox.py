"""
testing ground :)
"""
import torch
from datasets import load_dataset
from transformers import ClapModel, ClapProcessor

librispeech_dummy = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
audio_sample = librispeech_dummy[0]

model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech").to(0)
processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")

inputs = processor(audios=audio_sample["audio"]["array"], return_tensors="pt").to(0)

audio_embed = model.get_audio_features(**inputs)
