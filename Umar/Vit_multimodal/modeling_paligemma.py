from typing import Dict, List, Optional, Tuple, Union, Iterable
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from PIL import Image

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    # Explained: 1st image_token is the 128 token that describes the image 
    #            2nd: image_seq_len is the number of tokens that describe the image
    #            3rd: bos_token is the beginning of the sentence token 
    #            4th: prefix_prompt is the text that the prompt 

    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"  # \n charater is used to indicate the end of the sentence to the target prompt

        

def resize(
        image: Image,
        size: Tuple[int, int], 
        resample: Image.Resampling = None,
        reducing_gap: Optional[float] = None,
    )-> np.ndarray:
    """
    Resize the image to the specified size
    """
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image

def rescale(
        image: np.ndarray, 
        scale: float, 
        dtype: np.dtype = np.float32,
    ) -> np.ndarray:
    """
    Rescale the pixel values of the image to the range [0, 1]
    """
    rescaled_image = image.astype(dtype) * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(
        image: np.ndarray,
        mean: Union[float, Iterable[float]],
        std: Union[float, Iterable[float]],
    ) -> np.ndarray:
    """
    Normalize the pixel values of the image to have a mean of 0 and standard deviation of 1
    """
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    normalized_image = (image - mean) / std
    return normalized_image

def process_image(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """
    Preprocess the images to the format required by the model
    """
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    # COnvert each image to a numpy array
    images = [np.array(image) for image in images]
    # Resacele the pixel values to the range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the pixel values to have a mean of 0 and standard deviation of 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Move the channel dimension to the 1st dimension. The model expects the channel dimension in the foramt [channel, height, width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images


# PALIGEMMA: input: image, text, output: image, text
# it can be segmentation and object detection, image captioning, image generation, etc.
class PaligemmaProcessor(nn.Module):

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):  # This happens when you call the class like this: PaligemmaProcessor(tokenizer, num_image_tokens, image_size) for starting the program
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

    def __call__(# This happens when you call the class like this: processor =   PaligemmaProcess(image, ext)
        # Original model can do also object detection, image captioning, image generation, etc. but here we ust only image captioning and text generation
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True, 
        ) -> dict:
        
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts. Only single image and text pairs are supported"

        pixel_values = process_image(   # since paligamma acceprts image of the size 224x224, we need to resize the image
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1.0 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        # Convert the list of numpy arrays to a single numpy array with shape [batch_size, channels, height, width]
        pixel_values = np.stack(pixel_values, axis=0)   # Converted to one single tensor
        pixel_values = torch.tensor(pixel_values, dtype=torch.float32)

        # Prepend a 'self.image_seq_length' number of image tokens to the input
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len = self.image_seq_len,
                image_token = self.IMAGE_TOKEN,
            )
            for prompt in text
        
        ]

        # Return the input_ids and attention_mask as pytorch tensors
        inputs = self.tokenizer(
            input_strings,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data

