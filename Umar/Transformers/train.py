import torch
import torch.nn as nn

from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


def get_all_sentences(ds, lang):
    # ds: dataset
    # lang: language selected

    get item in ds:
        for item in ds:
            yield item['translation'][lang]    


def get_or_build_tokenizer(config, ds, lang):
    # Config: config file
    # ds: dataset
    # lang: language selected

    tokenizer_path = Path(config{"tokenizer_path"}.format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))  # a worf which is not in the vocabulary will be replaced by [UNK]
        tokenizer.pre_tokenizer = Whitespace()              # tokenizes the text by splitting on whitespace
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]","[SOS]", "[EOS]"], min_frequency=2) 
            # it has special tokens and min_frequency of 2.
            # Speical tokens:  [UNK]: unknown token, [PAD]: padding token, [SOS]: start of sentence, [EOS]: end of sentence
            # Min_frequency: 2 -> the word should appear atleast 2 times in the dataset to be included in the vocabulary

    # Training the tokenizer
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    # Config: config file

    dataset = load_dataset("Helsinki-NLP/opus_books", split="train", f'{config["lang_src"]}-{config["lang_tgt"]}')
    # have to check the usage of the above line
    
    # build tokenizers

