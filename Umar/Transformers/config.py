def get_config():
    return {
        "batch_size": 9,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,  # different for each dataset
        "d_model": 512, 
        "lang_src": "en",
        "lang_src": "it",
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload": None, 
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tomodel"
    }

def get_weigths_file_path(config, epoch:str):
    