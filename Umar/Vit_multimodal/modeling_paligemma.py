from typing import Dict, List, Optional, Tuple, Union, Iterable
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from PIL import Image

# PALIGEMMA: input: image, text, output: image, text
# it can be segmentation and object detection, image captioning, image generation, etc.
class PaligemmaProcessor(nn.Module):

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.image_seq_len = num_image_tokens
        self.image_size = image_size

        # Tokenizer descirbes as here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        # https://huggingface.co/blog/paligemma
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}   # Token for image
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for object segmentation
        tokens_to_add = {"additional_special_tokens": EXTRA_TOKENS}
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # we will add BOS and EOS tokens to the image tokens
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer  

    def __call__(self, image: Image.Image, text: str):