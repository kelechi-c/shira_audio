from typing import Literal
from .utils import audiofile_crawler, read_audio, latency
from datasets import load_dataset, Dataset
from transformers import ClapPreTrainedModel, ClapProcessor, ClapAudioModel, ClapTextModel


class AudioSearch:
    def __init__(self):
        pass
    
    def wavesearch(self):
        pass


# indexes the audio files for text/audio-based retrieval
class AudioEmbedding:
    def __init__(
        self, 
        data_path: str, 
        embed_model_id: str = "laion/larger_clap_music_and_speech",
        dataset_type: Literal['huggingface', 'local_folder'] = 'huggingface', 
        device: Literal['cuda', 'cpu'] = 'cpu',
    ):
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.model_id = embed_model_id
        self.device = device
        self.audio_dataset = None

        # load models for processing/retrieval
        self.processor = ClapProcessor.from_pretrained(self.model_id)
        self.embed_model = ClapAudioModel.from_pretrained(self.model_id)
        print(f"loaded CLAP model/processor _{self.model_id}")
        
        if dataset_type == 'huggingface':
            self.audio_dataset = load_dataset(data_path, split='validation', trust_remote_code=True)
        else:
            audiofiles = audiofile_crawler(data_path)
            self.audio_dataset = Dataset.from_dict({'audio': audiofiles})

    @latency
    def index_files(self):
        pass 