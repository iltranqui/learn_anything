import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console

from dataset import BilingualDataset, casual_mask
import numpy as np
from model import build_transform
from datasets import load_dataset

from config import get_weigths_file_path, get_config
from tqdm import tqdm

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
import pytest

import pretty_errors
import warnings # to avoid reading all the warnings
console = Console()

"""
    What does an encoding do ? -> it divides the text into tokens which can be handled
    output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
    print(output.tokens)
    # ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]
"""

"""
    From Words (Paragraph and Sentences) to Embeddings(Words):
        Step 1: Divide all sentences -> we get individual sentences like they were images. Each sentences is a unit basically. [ like we can have bathces of 8 sentences like habing a batch of 8 images ]
        Step 2: Encode the sentences -> Like above, we divide the sentences into words and then each word into tokens -> Token is about representing a word with a number
        OBS: following steps are done by transformers
        Step 3: Embedding the tokens -> we amplify the token number ( which represents a word or more word) into a vector of 512 numbers, to represent different uses of the word, like an emotion or situation
"""


def get_all_sentences(ds, lang):
    """
    Functions to get all the sentences from a torch.Dataset
    """
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
    # Embedding: a function to transform a number which represents a word, into a vector of 512

    # config['tokenizer_file' ]  = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

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

    dataset = load_dataset(f"{config['datasource']}", f'{config["lang_src"]}-{config["lang_tgt"]}', split="train", )
    # have to check the usage of the above line

    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, dataset, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset, config['lang_tgt'])

    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(dataset))
    #val_ds_size = int(0.1 * len(dataset))  -> SOlve the problem of random_split
    val_ds_size = len(dataset) - train_ds_size
    # print(f"TRAIN LEN: {train_ds_size}")
    # print(f"VAL LEN: {val_ds_size}")
    # print(f"DATASET LEGNTRH: {len(dataset)}")
    train_ds_raw, val_ds_raw = random_split(dataset, [train_ds_size, val_ds_size])


    # Loading the dataset
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])



    # Verify the max length for the tgt and src sentences so
    max_len_src = 0
    max_len_tgt = 0

    for item in dataset:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    # print(f"Max Length of source sentence: {max_len_src}")
    # print(f"Max length of target sentences; {max_len_tgt}")

    # Dataloaders

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt



def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transform(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model



# Training Loop #
def translate_sentence(model, sentence, tokenizer_src, tokenizer_tgt, device, max_len=350):
    """Translate a sentence from source language to target language"""
    # Set the model to evaluation mode
    model.eval()

    # Tokenize the source sentence
    tokens = ["[SOS]"] + tokenizer_src.encode(sentence).tokens + ["[EOS]"]
    src_indexes = [tokenizer_src.token_to_id(token) for token in tokens]

    # Add padding if needed
    if len(src_indexes) < max_len:
        src_indexes = src_indexes + [tokenizer_src.token_to_id("[PAD]")] * (max_len - len(src_indexes))
    else:
        src_indexes = src_indexes[:max_len-1] + [tokenizer_src.token_to_id("[EOS]")]

    # Convert to tensor and add batch dimension
    encoder_input = torch.tensor([src_indexes]).to(device)

    # Create source mask
    src_mask = (encoder_input != tokenizer_src.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(2).to(device)

    # Encode the source sentence
    with torch.no_grad():
        encoder_output = model.encode(encoder_input, src_mask)
        if encoder_output is None:
            # If encoder output is None, just return a placeholder message
            return "[Translation failed - model not ready yet]"

    # Initialize the decoder input with SOS token
    decoder_input = torch.tensor([[tokenizer_tgt.token_to_id("[SOS]")]]).to(device)

    # Generate the translation one token at a time
    output_sequence = []

    for _ in range(max_len):
        # Create target mask
        tgt_mask = casual_mask(decoder_input.size(1)).to(device)

        # Decode the encoder output
        with torch.no_grad():
            decoder_output = model.decode(encoder_output, src_mask, decoder_input, tgt_mask)
            if decoder_output is None:
                # If decoder output is None, just return a placeholder message
                return "[Translation failed - model not ready yet]"

            projection = model.project(decoder_output)
            if projection is None:
                # If projection is None, just return a placeholder message
                return "[Translation failed - model not ready yet]"

            # Get the next token
            next_token_logits = projection[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=1).item()

            # Add the token to the output sequence
            output_sequence.append(next_token)

            # Break if EOS token is generated
            if next_token == tokenizer_tgt.token_to_id("[EOS]"):
                break

            # Update decoder input for next iteration
            decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]]).to(device)], dim=1)

    # Convert token IDs to tokens
    output_tokens = [tokenizer_tgt.id_to_token(idx) for idx in output_sequence if idx not in
                    [tokenizer_tgt.token_to_id("[PAD]"), tokenizer_tgt.token_to_id("[EOS]")]]

    # Join tokens to form the translated sentence
    translated_sentence = tokenizer_tgt.decode(output_tokens)

    return translated_sentence


def train_model(config):
    # Define the device to put on the tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)

    # Create weights folder
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # in case the model crashes during training
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weigths_file_path(config, config['preload'])
        # print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']


    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)   # tell him to ignore the padding

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) #   ( B , Seq_len )
            decoder_input = batch['decoder_input'].to(device) # ( b, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) #   ( B , 1, 1, Seq_len )
            decoder_mask = batch['decoder_mask'].to(device) # ( b, 1, Seq_len, seq_len)

            # Run the tensor through he transformer
            encoder_output = model.encode(encoder_input, encoder_mask)   # ( B, Seq_LEn, d_model)

            # Skip this batch if encoder output is None
            if encoder_output is None:
                # print("Skipping batch due to None encoder output")
                continue

            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (B, Seq_Len, d_model)

            # Skip this batch if decoder output is None
            if decoder_output is None:
                # print("Skipping batch due to None decoder output")
                continue

            # Projection Output
            proj_output = model.project(decoder_output) # ( B, Seq_Len)

            # Skip this batch if projection output is None
            if proj_output is None:
                # print("Skipping batch due to None projection output")
                continue

            # THis transforms ( B , Seq_Len, tgt_vocab_size )  ---> ( B * Seq_Len, tgt_vocab_size )
            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the Loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            #Update the weights
            optimizer.step()   # Does the backpropagation
            optimizer.zero_grad()

            global_step += 1

        # Save the model at the end of every epoch
        model_filename = get_weigths_file_path(config, f"{epoch:02d}")

        # it is best not only to save the model at each epoch, you need to save also the states of the optimizer
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

        # Run inference on the test sentence after every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == config['num_epochs'] - 1:
            # Test sentence: "Hello, y'all! How are you [UNK] ?"
            test_sentence = "Hello, y'all! How are you ?"
            print(f"\nEpoch {epoch+1} - Translating test sentence: '{test_sentence}'")
            translated = translate_sentence(model, test_sentence, tokenizer_src, tokenizer_tgt, device)
            print(f"Translation: '{translated}'\n")

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    pytest.main(['-v', 'test_transformer.py'])  # Run the tests
    train_model(config)

