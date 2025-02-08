import os, sys
import torch
from multiprocessing import freeze_support
sys.path.append('/home/yeok/projects/xtr-warp')
from warp.engine.config import WARPRunConfig
from warp.engine.searcher import WARPSearcher
from warp.engine.utils.collection_indexer import index
from warp.engine.utils.index_converter import convert_index

class CustomWARPRunConfig(WARPRunConfig):
    def __init__(self, collection, nbits: int = 4, k: int = 10):
        self.collection = collection
        self.nbits = nbits
        self.k = k

    @property
    def experiment_name(self):
        return f"{self.collection.name}"

    @property
    def index_name(self):
        return f"{self.collection.name}.nbits={self.nbits}"

    @property
    def collection_path(self):
        return self.collection.path

passages = [
    "Bananas are berries, but strawberries aren't.",
    "Octopuses have three hearts and blue blood.",
    "A day on Venus is longer than a year on Venus.",
    "There are more trees on Earth than stars in the Milky Way.",
    "Water can boil and freeze at the same time, known as the triple point.",
    "A shrimp's heart is located in its head.",
    "Honey never spoils; archaeologists have found 3000-year-old edible honey.",
    "Wombat poop is cube-shaped to prevent it from rolling away.",
    "There's a species of jellyfish that is biologically immortal.",
    "Humans share about 60% of their DNA with bananas.",
    "The Eiffel Tower can grow taller in the summer due to heat expansion.",
    "Some turtles can breathe through their butts.",
    "The shortest war in history lasted 38 to 45 minutes (Anglo-Zanzibar War).",
    "There's a gas cloud in space that smells like rum and tastes like raspberries.",
    "Cows have best friends and get stressed when separated.",
    "A group of flamingos is called a 'flamboyance'.",
    "There's a species of fungus that can turn ants into zombies.",
    "Sharks existed before trees.",
    "Scotland has 421 words for 'snow'.",
    "Hot water freezes faster than cold water, known as the Mpemba effect.",
    "The inventor of the frisbee was turned into a frisbee after he died.",
    "There's an island in Japan where bunnies outnumber people.",
    "Sloths can hold their breath longer than dolphins.",
    "You can hear a blue whale's heartbeat from over 2 miles away.",
    "Butterflies can taste with their feet.",
    "A day on Earth was once only 6 hours long in the distant past.",
    "Vatican City has the highest crime rate per capita due to its tiny population.",
    "There's an official Wizard of New Zealand, appointed by the government.",
    "A bolt of lightning is five times hotter than the surface of the sun.",
    "The letter 'E' is the most common letter in the English language.",
    "There's a lake in Australia that stays bright pink regardless of conditions.",
    "Cleopatra lived closer in time to the first moon landing than to the building of the Great Pyramid."
]

class CustomCollection:
    def __init__(self, name: str, path: str, passages):
        self.name = name
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as file:
            file.writelines([f"{pid}\t{passage}\n" for pid, passage in enumerate(passages)])

def construct_index(config: WARPRunConfig):
    index(config)
    convert_index(os.path.join(config.index_root, config.index_name))

def print_query(searcher: WARPSearcher, query: str):
    print(f"Query: {query}")
    passage_ids, _, scores = searcher.search(query, k=10)
    for pid, score in zip(passage_ids, scores):
        print(pid, passages[pid], score)
    print("====================")

if __name__ == '__main__':
    os.environ["INDEX_ROOT"] = os.path.expanduser("~/warp/indexes")
    freeze_support()
    torch.set_num_threads(1)

    # Define the collection (i.e., list of passages)
    collection_path = os.path.expanduser("~/collections/research.tsv")
    collection = CustomCollection(
        name="research",
        path=collection_path,
        passages=passages
    )
    config = CustomWARPRunConfig(
        nbits=4,
        collection=collection,
    )

    # Construct an index over the provided collection.
    construct_index(config)

    # Prepare for searching via the constructed index.
    searcher = WARPSearcher(config)

    # Handle "user" queries using the searcher.
    print_query(searcher, "how do butterflies taste?")
    print_query(searcher, "quickest war in history?")