## shira 🔖🎧

### A simple audio search/retrieval library. (wip)

This is the audio version of [**ripple**](https://github.com/kelechi-c/ripple_net).
**Search** through **audio files/data** with **text queries** or **audio samples**.\
It's meant to be an **_neural_ encoded** version of [Shazam](https://www.shazam.com/), but might just be for small scale/local usage.

#### Methodology
It's basically a **semantic search library for audio**.

The local audio data/files are indexed and embeddings are generated(with CLAP), 
then a **FAISS** vector index is created.\
The files are retrieved based on **cosine similarity** between embeddings.
(it could also be adapted for **audio recommendation**).

This process makes use of contrastively pretrained audio-language model, **CLAP**(like **OpenAI CLIP** for audio), 
specifically LAION's **[laion/larger_clap_music_and_speech](https://huggingface.co/laion/larger_clap_music_and_speech)** checkpoint/model

<!-- #### general info
#### usage -->
#### usage
- Install the library

```bash
pip install shira-audio
```
- **For text-based search**
```python
from shira import AudioSearch, AudioEmbedding

embedder = AudioEmbedding(data_path='.') # init embedder class
audio_data_embeds = embedder.index_files() # create embeddings and index audio files

neural_search = AudioSearch() # init semantic search class

text_query = 'classical music' # text description for search

# get k similar audio w/probability score pairs 
matching_samples, scores = neural_search.text_search(text_query, audio_data_embeds, k_count=5)

matching_samples['path'][0], scores[0] # get file path for the top sample
```

Or you could use it from your terminal:
```bash
# -t for text query 
# --dir for [optional] target directory 
shira_text -t instrumental --dir downloads/music
```

#### Acknowldgements
- [**CLAP**: Learning Audio Concepts From Natural Language Supervision](https://arxiv.org/abs/2206.04769)
- [**tinyCLAP**: Distilling Contrastive Language-Audio Pretrained models]() 
- [Large-scale **contrastive language-audio pretraining** with feature fusion and keyword-to-caption augmentation](https://arxiv.org/abs/2211.06687)
- [**laion/larger_clap_music_and_speech**](https://huggingface.co/laion/larger_clap_music_and_speech) model by LAION.