from pathlib import Path
import click
from .audio_search import AudioEmbedding, AudioSearch
from .utils import audiofile_crawler


@click.command()
@click.option('-d', '--dir', default=lambda: Path.home(), type=str)
def check_files(dir):
    _, f_count = audiofile_crawler(dir)
    click.echo('file crawling complete')


@click.command()
@click.option("-t", "--textquery", required=True, type=str, help="text description for audio retrieval")
@click.option("-d", "--dir", default=lambda: Path.home(), type=str, help='target directory')
def text_search(textquery: str, dir):
    
    embedder = AudioEmbedding(data_path=dir) # init embedder class
    
    audio_data_embeds = embedder.index_files() # create embeddings and index audio files
    neural_search = AudioSearch() # init semantic search class

    # get k similar audio w/probability scores pairs
    matching_samples, scores = neural_search.text_search(textquery, audio_data_embeds, k_count=5) # type: ignore

    top_sample = matching_samples[0]['audio']['path'] # get file path for top sample
    score = scores[0] * 100
    click.echo(f"text query {textquery}")
    click.echo("...........")
    click.echo(f"search result #1 {top_sample}, p = {score}%")