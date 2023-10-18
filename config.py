from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 1024,
        "token_to_word_ratio": 0.75,
        "d_model": 512,
        "datasource": "iitb-english-hindi",
        "lang_src": "en",
        "lang_tgt": "hi",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)