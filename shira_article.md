# shira: neural audio search/recommendation with CLAP

### overview
**Hello there, it's Tensor**

In this article, I will give a walkthrough on how I built this library/system, code samples, intuition, etc.
First off, I have always admired [**Shazam**](https://www.shazam.com/), so I decided to reproduce 
the workings with neural networks. 
So in the process, I built this library for audio retrieval, semantic search and possibly recommendation, which I named **shira**. THe code is open-sourced at [**https://github.com/kelechi-c/shira_audio**](https://github.com/kelechi-c/shira_audio).

### So how does shira even work?
**shira** uses a pretrained **CLAP** model, specifically LAION's **[laion/larger_clap_music_and_speech](https://huggingface.co/laion/larger_clap_music_and_speech)**. The local audio files/dataset is indexed and embeddings are generated (with CLAP's audio encoder), 
then a **FAISS** vector embedding index is created.\
The files are retrieved based on **cosine similarity** between embeddings.

Since in **contrastive text-audio pretraining**, 
the text and audio vectors are projected to a **joint embedding space**, this enables cross-modal retrieval,
meaning both text/audio can be used to retrieve **semantically similar audio**.

![clap model framework](/assets/clap_structure.png)

For more info on **CLAP** check out the [**paper on arxiv**](https://arxiv.org/abs/2211.06687)

### what does this look like in code?