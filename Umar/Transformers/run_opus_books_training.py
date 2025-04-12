import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import warnings

from dataset import BilingualDataset, casual_mask
from model import build_transform
from datasets import load_dataset
from config import get_weigths_file_path, get_config
from tqdm import tqdm

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

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

def get_dataset(config):
    """
    Load and prepare the dataset
    """
    # Load the dataset
    dataset = load_dataset(f"{config['hugging_face']}", f"{config['lang_src']}-{config['lang_tgt']}", split="train")
    
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, dataset, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset, config['lang_tgt'])
    
    # Split dataset
    train_ds_size = int(0.9 * len(dataset))
    val_ds_size = len(dataset) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(dataset, [train_ds_size, val_ds_size])
    
    # Create bilingual datasets
    train_ds = BilingualDataset(
        train_ds_raw, 
        tokenizer_src, 
        tokenizer_tgt, 
        config['lang_src'], 
        config['lang_tgt'], 
        config['seq_len']
    )
    
    val_ds = BilingualDataset(
        val_ds_raw, 
        tokenizer_src, 
        tokenizer_tgt, 
        config['lang_src'], 
        config['lang_tgt'], 
        config['seq_len']
    )
    
    # Verify the max length for the tgt and src sentences
    max_len_src = 0
    max_len_tgt = 0
    
    for item in dataset:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f"Max Length of source sentence: {max_len_src}")
    print(f"Max length of target sentences: {max_len_tgt}")
    
    # Create dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    Create the transformer model
    """
    model = build_transform(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    """
    Train the transformer model
    """
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create weights folder
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    # Get dataset and model
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    # In case the model crashes during training
    initial_epoch = 0
    global_step = 0
    
    if config['preload']:
        model_filename = get_weigths_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    
    # Training loop
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            
            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            
            # Projection output
            proj_output = model.project(decoder_output)
            
            # Calculate loss
            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            
            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            
            # Backpropagate the loss
            loss.backward()
            
            # Update the weights
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
        
        # Save the model at the end of every epoch
        model_filename = get_weigths_file_path(config, f"{epoch:02d}")
        
        # Save model and optimizer state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        
        print(f"Model saved as {model_filename}")

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
