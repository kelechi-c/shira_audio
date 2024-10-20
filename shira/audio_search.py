import os
import numpy as np
from typing import Literal, Union
from pathlib import Path
from .utils import audiofile_crawler, filter_files, latency, read_audio
from datasets import load_dataset, Dataset, DatasetDict, Audio
from transformers import ClapModel, ClapProcessor

# local path variables
LOCAL_MODEL_PATH = Path.home()/'clap_model'
LOCAL_PROCESSOR_PATH = Path.home()/'clap_processor'
LOCAL_DATA_EMBED = Path.home()/'audio_embeddings'


class AudioSearch:
    def __init__(
        self,
        model_id: str = "laion/larger_clap_music_and_speech",
        local_model_path: Union[str, Path] = LOCAL_MODEL_PATH,
        local_processor_path: Union[str, Path] = LOCAL_PROCESSOR_PATH,
        embedding_path: Union[str, Path] = LOCAL_DATA_EMBED,
        device: str = 'cpu'
    ):
        self.model_id = model_id
        self.local_model_path = local_model_path
        self.local_processor_path = local_processor_path
        self.embeds_path = embedding_path
        self.device = device
        self.processor = None
        self.clap_model = None

        # load models for semantic search retrieval
        self.clap_model, self.processor = load_models(local_model_path, local_processor_path, model_id)

    @latency
    def text_search(
        # semantic searhc with text description
        self, 
        text_query: str, 
        embedded_data: Union[Dataset, DatasetDict],
        k_count: int = 4, 
        device: str = 'cpu'
    ):

        # preprocess and tokenize text
        encoded_text = self.processor(text=text_query, return_tensors="pt")["input_ids"]  # type: ignore
        encoded_text = encoded_text.to(device) # type: ignore

        # get aligned text embeddings for query
        text_embed = self.clap_model.get_text_features(encoded_text)[0] # type: ignore
        text_embed = text_embed.detach().cpu().numpy()

        # get nearest samples based on faiss mebeddings
        scores, retrieved_audio = embedded_data.get_nearest_examples("audio_embeddings", text_embed, k=k_count) # type: ignore

        return retrieved_audio, scores

    @latency 
    def audio_search(
        # basically neural Shazam (and a bit slow)
        # matching/similar audio models
        self,
        input_audio: Union[str, os.PathLike],
        embedded_data,
        k_count: int = 2,
        device: str = "cpu",
    ):
        if not isinstance(input_audio, np.ndarray):  
            input_audio = read_audio(input_audio)  # type: ignore # loads audio file to ndarray

        audio_values = self.processor(audios=input_audio, return_tensors="pt", sampling_rate=sample_rate)["audio_features"] # type: ignore
        audio_values = audio_values.to(device) # type: ignore

        wav_embed = self.clap_model.get_audio_features(audio_values)[0] # type: ignore
        wav_embed = wav_embed.detach().cpu().numpy()

        scores, retrieved_audio = embedded_data.get_nearest_examples("audio_embeddings", wav_embed, k=k_count)

        return retrieved_audio, scores


# indexes the audio files for text/audio-based retrieval
class AudioEmbedding:
    def __init__(
        self,
        data_path: Union[str, Path] = Path.home(), # load from home directory if not specified
        embed_model_id: str = "laion/larger_clap_music_and_speech", # clap model id, use LAION checkpoint if not specified
        dataset_type: Literal['huggingface', 'local_folder'] = 'local_folder', # load from directory or remote repo
        device: Literal['cuda', 'cpu'] = 'cpu', # for GPU(faster) or CPU usage
        audio_embed_path:  Union[str, Path] = LOCAL_DATA_EMBED
    ):
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.model_id = embed_model_id
        self.model_save_path = LOCAL_MODEL_PATH
        self.processor_path = LOCAL_PROCESSOR_PATH
        self.audio_embed_path = audio_embed_path
        self.device = device
        self.audio_dataset = None
        self.processor = None
        self.embed_model = None

        # load models for processing/retrieval
        self.embed_model, self.processor = load_models(self.model_save_path, self.processor_path, self.model_id)

        # load dataset from remote repo of local audio files
        if dataset_type == 'huggingface':
            self.audio_dataset = load_dataset(data_path, split='train', trust_remote_code=True) # type: ignore
        else:
            audiofiles, file_count = audiofile_crawler(data_path) # get all audio files under the directory
            self.audio_dataset = Dataset.from_dict({'audio': audiofiles, 'path': audiofiles})
            self.audio_dataset = self.audio_dataset.cast_column('audio', Audio(sampling_rate=22400))

    @latency
    def index_files(self) -> Union[Dataset, DatasetDict]: # create faiss index for audio files
        assert self.device in ["cuda", "cpu",], "Wrong device id, must either be 'cuda' or 'cpu'"

        # encode/embed arrays for search
        embedded_data = self.audio_dataset.map(self.embed_audio_batch) # type: ignore
        embedded_data.add_faiss_index(column="audio_embeddings")  # type: ignore # Initialize FAISS index
        # embedded_data.save_to_disk(str(self.audio_data_embed))
        print(f"created faiss vector embeddings/index for {self.data_path} @ {self.audio_embed_path}")

        return embedded_data # type: ignore

    def embed_audio_batch(self, batch): # encode audio and add 'embeddings' column for indexing
        sample = batch["audio"]['array']

        try:
            # preprocessing/normalization with CLAP audio processor
            coded_audio = self.processor(
                audios=sample, 
                return_tensors="pt", 
                sampling_rate=48000
            )["input_features"] # type: ignore

            # feature extraction and embedding projection
            audio_embed = self.embed_model.get_audio_features(coded_audio) # type: ignore

            batch["audio_embeddings"] = audio_embed[0]

            return batch

        except Exception as e:
            print(f'error in file processing: {e}')


def load_models(
    local_model_path: Union[str, os.PathLike],
    local_processor_path: Union[str, os.PathLike],
    model_id: str,
):
    clap_model = None
    processor = None

    is_local = os.path.isdir(
        local_model_path
    )  # check if previously saved models are available

    if is_local:  # load from locally saved weights
        clap_model = ClapModel.from_pretrained(local_model_path)
        processor = ClapProcessor.from_pretrained(local_processor_path)
        print(f"loaded retrieval model from local storage @ {local_model_path}")

    else:  # download fresh weights from huggingface
        clap_model = ClapModel.from_pretrained(model_id)
        processor = ClapProcessor.from_pretrained(model_id)
        print(f'loaded {model_id} weights from huggingface')

        # save model locally so you don't have to download 1gb+ weights everytime
        clap_model.save_pretrained(local_model_path)
        processor.save_pretrained(local_processor_path) # type: ignore
        print(f'saved model @ {local_model_path}')

    return clap_model, processor
