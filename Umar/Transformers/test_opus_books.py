import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import DataLoader, random_split

from dataset import BilingualDataset
from config import get_config

def get_all_sentences(ds, lang):
    """
    Functions to get all the sentences from a torch.Dataset
    """
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """
    Get or build a tokenizer for the specified language
    """
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]","[SOS]", "[EOS]"], min_frequency=2)

        # Training the tokenizer
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def test_opus_books_dataset():
    """
    Test loading and processing the Opus Books dataset
    """
    config = get_config()
    
    # Load the dataset
    print(f"Loading dataset: {config['datasource']} with language pair {config['lang_src']}-{config['lang_tgt']}")
    try:
        dataset = load_dataset(f"{config['hugging_face']}", f"{config['lang_src']}-{config['lang_tgt']}", split="train")
        print(f"Dataset loaded successfully with {len(dataset)} examples")
        
        # Display a few examples
        print("\nSample examples from the dataset:")
        for i in range(min(3, len(dataset))):
            print(f"Example {i+1}:")
            print(f"Source ({config['lang_src']}): {dataset[i]['translation'][config['lang_src']]}")
            print(f"Target ({config['lang_tgt']}): {dataset[i]['translation'][config['lang_tgt']]}")
            print()
        
        # Build tokenizers
        print("Building tokenizers...")
        tokenizer_src = get_or_build_tokenizer(config, dataset, config['lang_src'])
        tokenizer_tgt = get_or_build_tokenizer(config, dataset, config['lang_tgt'])
        
        print(f"Source tokenizer vocabulary size: {tokenizer_src.get_vocab_size()}")
        print(f"Target tokenizer vocabulary size: {tokenizer_tgt.get_vocab_size()}")
        
        # Split dataset
        train_ds_size = int(0.9 * len(dataset))
        val_ds_size = len(dataset) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(dataset, [train_ds_size, val_ds_size])
        
        print(f"Training set size: {train_ds_size}")
        print(f"Validation set size: {val_ds_size}")
        
        # Create bilingual datasets
        train_ds = BilingualDataset(
            train_ds_raw, 
            tokenizer_src, 
            tokenizer_tgt, 
            config['lang_src'], 
            config['lang_tgt'], 
            config['seq_len']
        )
        
        # Test creating a batch
        train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        
        print("\nTesting batch creation...")
        for batch in train_dataloader:
            print(f"Encoder input shape: {batch['encoder_input'].shape}")
            print(f"Decoder input shape: {batch['decoder_input'].shape}")
            print(f"Encoder mask shape: {batch['encoder_mask'].shape}")
            print(f"Decoder mask shape: {batch['decoder_mask'].shape}")
            print(f"Label shape: {batch['label'].shape}")
            
            # Print a sample source and target text
            print(f"\nSample source text: {batch['src_text'][0]}")
            print(f"Sample target text: {batch['tgt_text'][0]}")
            break
        
        print("\nTest completed successfully!")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_opus_books_dataset()
