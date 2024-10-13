"""
testing ground :)
"""
import torch
import numpy as np
from datasets import load_dataset, Audio
from transformers import ClapModel, ClapProcessor
from src.shira.utils import read_audio, audiofile_crawler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "laion/larger_clap_music_and_speech"
sample_rate = 48000
max_duration = 10
batch_size = 16  # for batched mapping
input_path = "."
audiofiles = audiofile_crawler(input_path)

music_data = load_dataset("audiofolder", data_files=audiofiles, split="train").cast_column('audio', Audio(sampling_rate=sample_rate))
# music_data = Dataset.from_dict({'audio': [audiofiles]})

clap_model = ClapModel.from_pretrained(model_id)#.to(device)
clap_processor = ClapProcessor.from_pretrained(model_id)


inputs = clap_processor(audios=audio_sample["audio"]["array"], return_tensors="pt").to(device) # type: ignore
audio_embed = clap_model.get_audio_features(**inputs)

print(audio_embed)


def embed_audio_batch(batch):
    sample = batch["audio"]["array"]
    coded_audio = clap_processor(
        audios=sample, return_tensors="pt", sampling_rate=48000
    )["input_features"] # type: ignore

    audio_embed = clap_model.get_audio_features(coded_audio)

    batch["audio_embeddings"] = audio_embed[0]

    return batch


embedded_data = music_data.map(embed_audio_batch)


embedded_data = sample_data.map(embed_image_batch, batched=True, batch_size=batch_size)
embedded_data.add_faiss_index("audio_embeddings")


@latency
def audio_search(
    input_audio, embedded_data, k_count: int = 2, device: torch.device = device
):
    if not isinstance(input_audio, np.ndarray):
        input_audio = read_audio(input_audio)  # loads audio file from wav to ndarray

    audio_values = clap_processor(audios=input_audio, return_tensors="pt", sampling_rate=sample_rate)["input_features"]  # type: ignore
    audio_values = audio_values.to(device)

    wav_embed = clap_model.get_audio_features(audio_values)[0]
    wav_embed = wav_embed.detach().cpu().numpy()

    scores, retrieved_audio = embedded_data.get_nearest_examples(
        "audio_embeddings", wav_embed, k=k_count
    )

    return retrieved_audio, scores


image = "beethoven.wav"
similar_images = audio_search(image, 4)  # search for similar audio files
