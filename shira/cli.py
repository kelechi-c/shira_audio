from pathlib import Path
import click
from .audio_search import AudioEmbedding, AudioSearch
from .utils import audiofile_crawler


@click.command() # to simply crawl and check for all the audio files under the directory
@click.option('-d', '--dir', default=lambda: Path.home(), type=str)
def check_files(dir):
    _, f_count = audiofile_crawler(dir)
    click.echo('audio file crawling complete')


@click.command() # semantic search with text description 
@click.option("-t", "--textquery", required=True, type=str, help="text description for audio retrieval")
@click.option("-d", "--dir", default=lambda: Path.home(), type=str, help='target directory')
def text_search(textquery: str, dir):

    embedder = AudioEmbedding(data_path=dir) # init embedder class

    audio_data_embeds = embedder.index_files() # create embeddings and index audio files
    neural_search = AudioSearch() # init semantic search class

    # get k similar audio w/probability scores pairs
    matching_samples, scores = neural_search.text_search(textquery, audio_data_embeds, k_count=5) # type: ignore
    top_samples = matching_samples['path']

    try:
        top_sample = top_samples[0] # get file path for top sample
    except:
        print('audio filepath not available')
        top_sample = top_samples['audio']

    click.echo(f"text query {textquery}")
    click.echo("...........")
    click.echo(f"search results =>")
    for idx, (res, score) in enumerate(zip(top_samples, scores)):
        click.echo(f"{idx}...{res}, p = {score}")


# cli function for audio reference search
@click.command()
@click.option(
    "-f",
    "--file",
    required=True,
    type=str,
    help="audio reference file for search",
)
@click.option("-d", "--dir", default=lambda: Path.home(), type=str, help="target audio directory")
def audio_search(file: str, dir):

    embedder = AudioEmbedding(data_path=dir) # init embedder class

    audio_data_embeds = (embedder.index_files())  # create embeddings and index audio files
    neural_search = AudioSearch()  # search class

    # get k similar audio w/probability scores pairs
    matching_samples, scores = neural_search.audio_search(file, audio_data_embeds, k_count=4)  # type: ignore
    top_samples = matching_samples["path"]

    click.echo(f"reference file {file}")
    click.echo("...........")
    click.echo(f"search results =>")
    for idx, (res, score) in enumerate(zip(top_samples, scores)):
        click.echo(f"{idx}...{res}, p = {score}")
