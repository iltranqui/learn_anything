import torch
import torch.nn as nn
from torch.utils.data import Dataset, Dataloader, random_split

from dataset import BilingualDataset, casual_mask

from model import build_transform

from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


def get_all_sentences(ds, lang):
    # ds: dataset
    # lang: language selected
    for item in ds:
        yield item['translation'][lang]    


def get_or_build_tokenizer(config, ds, lang):
    # Config: config file
    # ds: dataset
    # lang: language selected

    # Tokenizer: A function which transforms a word into a number 
        # Long and takes time to build one
    # Embedding: a function to trasnform a number which represents a word, into a vector of 512

    # config['tokenizer_file' ]  = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config{"tokenizer_path"}.format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))  # a word which is not in the vocabulary will be replaced by [UNK], this action is done by the tokenizer
        tokenizer.pre_tokenizer = Whitespace()               # tokenizer the text by splitting on whitespace
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]","[SOS]", "[EOS]"], min_frequency=2) 
            # it has special tokens and min_frequency of 2.
            # Special tokens:  [UNK]: unknown token, [PAD]: padding token, [SOS]: start of sentence, [EOS]: end of sentence
            # Min_frequency: 2 -> the word should appear at least 2 times in the dataset to be included in the vocabulary we are building

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
    tokenizer_src = get_or_build_tokenizer(config=config, ds=ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config=config, ds=ds_raw, config['lang_tgt']
                                           
    # Keep 90% for training and 10% for validation 
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = int(0.1 * len(ds_raw))
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])       


    # Loading the dataset
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt', config['seq_len']])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt', config['seq_len']])
    


    # Verify the max length for the tgt and src sentences so 
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    print(f"Max Length of source sentence: {max_len_src}")
    print(f"Max length of target sentences; {max_len_tgt}")

    # Datalodeers

    train_dataloader = Dataloader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = Dataloader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt



def get_model(config, vocab_src_len, vocab_tgt_len)
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model
