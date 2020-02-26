import os
import time
import glob
import random

from comet_ml import Experiment

import torch
import torch.optim as O
import torch.nn as nn
from torch.utils import data

from model import EncoderRNN, AttnDecoderRNN
from util import get_config, makedirs, get_filenames
from train import train, validate
from dataset import Dataset


if __name__ == '__main__':
    config = get_config()
    experiment = Experiment("wXwnV8LZOtVfxqnRxr65Lv7C2")
    experiment.log_parameters(config)
    if torch.cuda.is_available():
        torch.cuda.set_device(config["gpu"])
        device = torch.device('cuda:{}'.format(config["gpu"]))
    else:
        device = torch.device('cpu')
    
    number_of_examples = len(get_filenames(os.path.join(config["data"], "x")))
    list_ids = [str(i) for i in range(1, number_of_examples + 1)]
    random.shuffle(list_ids)
    t_ind, v_ind = round(number_of_examples * 0.7), round(number_of_examples * 0.9)
    train_indices, validation_indices, test_indices = list_ids[:t_ind], list_ids[t_ind:v_ind], list_ids[v_ind:]
    
    params = {'batch_size': config["training"]["batch_size"],
              'shuffle': config["training"]["shuffle"],
              'num_workers': config["training"]["num_workers"]}
    
    # Build generators
    training_set = Dataset(config["data"], train_indices)
    training_generator = data.DataLoader(training_set, **params)
    
    validation_set = Dataset(config["data"], validation_indices)
    validation_generator = data.DataLoader(validation_set, **params)
    
    test_set = Dataset(config["data"], test_indices)
    test_generator = data.DataLoader(test_set, **params)
    
    # double the number of cells for bidirectional networks
    # if config.birnn:
    #     config.n_cells *= 2
    
    if config["resume_snapshot"]:
        model = torch.load(config["resume_snapshot"], map_location=device)
    else:
        encoder = EncoderRNN(config["encoder"]).to(device)
        decoder = AttnDecoderRNN(config["decoder"]).to(device)
    
    criterion = nn.MSELoss()
    encoder_opt = O.SGD(encoder.parameters(), lr=config["encoder_optimizer"]["learning_rate"])
    decoder_opt = O.SGD(decoder.parameters(), lr=config["decoder_optimizer"]["learning_rate"])
    
    iterations = 0
    start = time.time()
    best_valid_loss = -1
    header = '  Time Epoch Iteration Progress    (%Epoch)   Loss'
    dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f}'.split(','))
    log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f}'.split(','))
    makedirs(config["result_directory"])
    print(header)
    
    with experiment.train():
        for epoch in range(config["training"]["epochs"]):
            for batch_idx, (X_batch, y_batch) in enumerate(training_generator):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                train_loss = train(X_batch, y_batch, encoder,
                                   decoder, encoder_opt, decoder_opt, criterion)
                experiment.log_metric("train_loss", train_loss, step=iterations)
                # checkpoint model periodically
                if iterations % config["every"]["save"] == 0:
                    snapshot_prefix = os.path.join(config["result_directory"], 'snapshot')
                    snapshot_path = snapshot_prefix + '_loss_{:.6f}_iter_{}_model.pt'.format(train_loss, iterations)
                    torch.save({
                        'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict(),
                        'encoder_opt': encoder_opt.state_dict(),
                        'decoder_opt': decoder_opt.state_dict(),
                    }, snapshot_path)
                    
                    for f in glob.glob(snapshot_prefix + '*'):
                        if f != snapshot_path:
                            os.remove(f)
                
                # evaluate performance on validation set periodically
                if iterations % config["every"]["validate"] == 0:
                    with experiment.validate():
                        valid_loss = 0
                        for X_batch_v, y_batch_v in validation_generator:
                            X_batch_v, y_batch_v = X_batch_v.to(device), y_batch_v.to(device)
                            valid_loss += validate(X_batch_v, y_batch_v, encoder, decoder, criterion)

                        experiment.log_metric("valid_loss", valid_loss, step=iterations)
                        print(dev_log_template.format(time.time()-start,
                                                      epoch, iterations, 1+batch_idx, len(training_generator),
                                                      100. * (1+batch_idx) / len(training_generator), train_loss, valid_loss))
                        
                        # update best valiation set accuracy
                        if valid_loss < best_valid_loss:
                            
                            # found a model with better validation set loss
                            best_valid_loss = valid_loss
                            snapshot_prefix = os.path.join(config["result_directory"], 'best_snapshot')
                            snapshot_path = snapshot_prefix + '_valloss_{}__iter_{}_model.pt'.format(valid_loss, iterations)
                            
                            # save model, delete previous 'best_snapshot' files
                            torch.save({
                                'encoder': encoder.state_dict(),
                                'decoder': decoder.state_dict(),
                                'encoder_opt': encoder_opt.state_dict(),
                                'decoder_opt': decoder_opt.state_dict(),
                            }, snapshot_path)
            
                            for f in glob.glob(snapshot_prefix + '*'):
                                if f != snapshot_path:
                                    os.remove(f)
                
                if iterations % config["every"]["log"] == 0:
                    # print progress message
                    print(log_template.format(time.time()-start,
                                              epoch, iterations, 1+batch_idx, len(training_generator),
                                              100. * (1+batch_idx) / len(training_generator), train_loss))
                iterations += 1

    with experiment.test():
        test_loss = 0
        for X_batch_t, y_batch_t in test_generator:
            X_batch_t, y_batch_t = X_batch_t.to(device), y_batch_t.to(device)
            test_loss += validate(X_batch_t, y_batch_t, encoder, decoder, criterion)
        experiment.log_metric("test loss", test_loss)
