"""
testing ground :)
"""
import torch
import numpy as np
from datasets import load_dataset
from transformers import ClapModel, ClapProcessor
from src.wavsearch.utils import read_audio

batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_id = "laion/larger_clap_music_and_speech"

sample_data = load_dataset("mahendra0203/musiccaps_processed", split="validation")
audio_sample = sample_data[0] # type: ignore

clap_model = ClapModel.from_pretrained(model_id)#.to(device)
clap_processor = ClapProcessor.from_pretrained(model_id)


inputs = clap_processor(audios=audio_sample["audio"]["array"], return_tensors="pt").to(device) # type: ignore
audio_embed = clap_model.get_audio_features(**inputs)

print(audio_embed)


def embed_image_batch(batch):
    coded_audio = clap_processor(audios=batch["audio"], return_tensors="pt")# type: ignore #["audio_values"]
    # coded_audio = coded_audio.to(device_id)
    print(f'coded audio shape {coded_audio}')
    audio_embed = clap_model.get_audio_features(**coded_audio)
    print(f"audio embeds shape {audio_embed}")

    batch["audio_embeddings"] = audio_embed

    return batch


embedded_data = sample_data.map(embed_image_batch, batched=True, batch_size=batch_size)
embedded_data.add_faiss_index("audio_embeddings")


def audio_search(input_audio, k_count: int, device: torch.device=device):
    if not isinstance(input_audio, np.ndarray):  # check if image type is PIL
        input_audio = read_audio(input_audio)  # loads pil image

    audio_values = clap_processor(audios=input_audio, return_tensors="pt")["audio_values"] # type: ignore
    audio_values = audio_values.to(device)
    
    wav_embed = clap_model.get_audio_features(audio_values)[0]
    wav_embed = wav_embed.detach().cpu().numpy()

    scores, retrieved_audio = embedded_data.get_nearest_examples(
        "audio_embeddings", wav_embed, k=k_count
    )

    return retrieved_audio, scores


image = "beethoven.wav"
similar_images = audio_search(image, 4)  # search for similar audio files
