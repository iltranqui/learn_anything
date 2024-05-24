from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,  # different for each dataset
        "d_model": 512, 
        "datasource": 'opus_books',
        "hugging_face": 'Helsinki-NLP/opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None, 
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tomodel"
    }

def get_weigths_file_path(config, epoch:str):
    model_folder = f"{config['datasource']}_{config('model_folder')}"
    model_basename = config('model_basename')
    model_filename = f"{config['model_basename']}{epoch}.pt" 
    return str(Path('-') / model_folder / model_filename)