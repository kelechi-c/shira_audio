import os
from typing import Literal
from .utils import audiofile_crawler, read_audio, latency
from datasets import load_dataset, Dataset
from transformers import ClapModel, ClapProcessor


class AudioSearch:
    def __init__(self):
        pass
    
    def wavesearch(self):
        pass


# indexes the audio files for text/audio-based retrieval
class AudioEmbedding:
    def __init__(
        self,
        data_path: str = '~', # load from home directory if not specfied
        embed_model_id: str = "laion/larger_clap_music_and_speech", # clap model id, use LAION checkpoint if not specified
        dataset_type: Literal['huggingface', 'local_folder'] = 'local_folder', # load from directory or remote repo
        device: Literal['cuda', 'cpu'] = 'cpu', # for GPU(faster) or CPU usage
        save_model: bool = True # whether to save the model
    ):
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.model_id = embed_model_id
        self.model_save_path = 'laion_clap'
        self.device = device
        self.audio_dataset = None

        # load models for processing/retrieval
        self.processor = ClapProcessor.from_pretrained(self.model_id)
        self.embed_model = ClapModel.from_pretrained(self.model_id)
        print(f"loaded CLAP model/processor _{self.model_id}")

        # load dataset from remote repo of local audio files
        if dataset_type == 'huggingface':
            self.audio_dataset = load_dataset(data_path, split='train', trust_remote_code=True)
        else:
            audiofiles = audiofile_crawler(data_path) # get all audio files under the directory
            self.audio_dataset = Dataset.from_dict({'audio': audiofiles})
            
        if save_model:
            # save model locally so you don't have to load everytime
            os.mkdir(self.model_save_path)
            self.embed_model.save_pretrained(self.model_save_path)
            self.processor.save_pretrained(self.model_save_path) # type: ignore
            print(f'model saved at {self.model_save_path}')

    @latency
    def index_files(self):
        assert self.device in ["cuda", "cpu",], "Wrong device id, must either be 'cuda' or 'cpu'"
        
        # encode/embed arrays for search
        embedded_data = self.audio_dataset.map(self._embed_audio_batch, batch_size=10, batched=True) # type: ignore
        print(f'created faiss vector embeddings for {self.data_path}')
        
        return embedded_data

    def _embed_audio_batch(self, batch):
        sample = batch["audio"]['array']
        
        coded_audio = self.processor(
            audios=sample, 
            return_tensors="pt", 
            sampling_rate=48000
        )["input_features"] # type: ignore

        audio_embed = self.embed_model.get_audio_features(coded_audio)

        batch["audio_embeddings"] = audio_embed[0]

        return batch
