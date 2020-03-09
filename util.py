import os
import json
import glob
import errno

import torch


def save_and_delete_snapshot(encoder, decoder, encoder_opt, decoder_opt, snapshot_path, snapshot_prefix):
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'encoder_opt': encoder_opt.state_dict(),
        'decoder_opt': decoder_opt.state_dict(),
    }, snapshot_path)
    for f in glob.glob(snapshot_prefix + '*'):
        if f != snapshot_path:
            os.remove(f)

def get_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config


def get_filenames(path):
    return os.listdir(path)


def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""
       
    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise
