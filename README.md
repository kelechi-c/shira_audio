## shira 🔖🎧

### A simple audio search/retrieval library. (wip)

This is the audio version of [ripple](https://github.com/kelechi-c/ripple_net).
It's meant to be an **_neural_ encoded** version of [Shazam](https://www.shazam.com/), but might just be for small scale/local usage.

#### Methodology
It's basically a **semantic search library for audio**.

The local audio data/files are indexed and embeddings are generated(with CLAP), 
then a **FAISS** vector index is created.\
The files are retrieved based on **cosine similarity** between embeddings.

This process makes use of contrastively pretrained audio-language model, **CLAP**(like **OpenAI CLIP** for audio), 
specifically LAION's **[laion/larger_clap_music_and_speech](https://huggingface.co/laion/larger_clap_music_and_speech)** checkpoint/model

<!-- #### general info
#### usage -->

#### Acknowldgements
- [**CLAP**: Learning Audio Concepts From Natural Language Supervision](https://arxiv.org/abs/2206.04769)
- [**tinyCLAP**: Distilling Contrastive Language-Audio Pretrained models]() 
- [Large-scale **contrastive language-audio pretraining** with feature fusion and keyword-to-caption augmentation](https://arxiv.org/abs/2211.06687)
- [**laion/larger_clap_music_and_speech**](https://huggingface.co/laion/larger_clap_music_and_speech) model by LAION.

<!-- - <a href="https://huggingface.co/fpaissan/tinyCLAP"> fpaissan/tinyCLAP: </a> distilled CLAP model by <a href="https://huggingface.co/fpaissan/">fpaissan </a> . -->


<!-- 
```bibtex
@misc{https://doi.org/10.48550/arxiv.2211.06687,
  doi = {10.48550/ARXIV.2211.06687},
  url = {https://arxiv.org/abs/2211.06687},
  author = {Wu, Yusong and Chen, Ke and Zhang, Tianyu and Hui, Yuchen and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  keywords = {Sound (cs.SD), Audio and Speech Processing (eess.AS), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  title = {Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
``` -->
