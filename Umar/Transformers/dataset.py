from typing import Any   #  is a special type int in Python that indicates that a function or method can accept any type of argument.
import torch 
import torch.nn as nn
from torch.utils.data import Dataset

# Buling a dataclass from Dataset from Huggingface, it will be used for the dataloader in the future 


class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # save the particular tokens into tensors like [UNK], [SOS], etc 
        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype=torch.int64)
        
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype=torch.int64) 
        
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype=torch.int64)   


    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:  # From dataset to the dataloader 
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # convert the text into tokens and then into id  
        # So divide each single sentence into single words
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids    # will be passed as arrays 
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids 

        # WE need to pad the sequence length, since not all sentences have the same length -> need to calucalte how many pads to add encoder size anddecode size  
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2   # the SOS and EOS tokens 
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1   # we add only the SOS to the encoder side and EOS to the decoder size 

        # number of padding takens should never be negative to ensure that 
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        

        # Build the 2 tensor for encoder input and decoder input, but also for the label 
            # 1st: input to decoder, 2nd: input to decoder, 3rd: output of the decoder
        
        encoder_input = torch.cat([
            self.sos_token,                                                                         # Start of Sentence Token 
            torch.Tensor(enc_input_tokens, dtype=torch.int64),                                     # Words as Tokens 
            self.eos_token,                                                                         # ENd of Sentence Token 
            torch.Tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)              # Number of padding required for the sentence, they will be ignored from the transformer
        ])

        decoder_input = torch.cat([
            self.sos_token,
            torch.Tensor(dec_input_tokensù, dtype=torch.int64),
            # ! self.eos_token, -> no eos token  
            torch.Tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        label = torch.cat([
            torch.Tensor(dec_input_tokens, dtype=int64),
            self.eos_token,
            torch.Tensor([self.pad_token] * dec_input_tokens, dtype=torch.int64)
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # ( 1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),   # (1, seq_lke)
            "label": label,   # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
        # Build a mask for the encoder

def casual_mask(size):
    # IN the self attention mask, we have a 2D matrix which a score of attention from each word in the sentence to each word 
    # since we are translating we want to look up only to the words who are before in the sentence and not after, so we will be exlcufing the upper triangle of the matrix
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).dtype=(torch.int)
    return mask == 0 
