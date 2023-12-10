import argparse
import torch.nn as nn
import numpy as np
import random
import os
import torch
import logging
import time 

from datetime import datetime
from torch.utils.data import DataLoader
from torch import optim

# from src.models.retnet import Retnet
from models import Model
from datasets import get_dataframe
from datasets import MyDataset
from metrics import f1_scores

def reduce_Lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group["lr"] / 100

def train(args, logger):

    device = "cpu" 
    if torch.cuda.is_available():
        device = "cuda"

    with open(os.path.join(args.path_data, 'genres.txt'), 'r') as f:
        genre_all = f.readlines()
        genre_all = [x.replace('\n','') for x in genre_all]
    
    genre2idx = {genre:idx for idx, genre in enumerate(genre_all)}
    
    """
    Load datasets
    """
    logger.info("Loading datasets!")
    print("Loading datasets!")
    movies_train = get_dataframe(os.path.join(args.path_data, "movies_train.csv"))
    movies_valid = get_dataframe(os.path.join(args.path_data, "movies_valid.csv"))
    movies_test = get_dataframe(os.path.join(args.path_data, "movies_test.csv"))

    train_datasets = MyDataset(movies_train, genre2idx)
    valid_datasets = MyDataset(movies_valid, genre2idx)
    test_datasets = MyDataset(movies_test, genre2idx)

    train_dataloader = DataLoader(train_datasets, batch_size=args.batch_size)
    valid_dataloader = DataLoader(valid_datasets, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_datasets, batch_size=args.batch_size)
    
    logger.info("Loaded dataset!")
    logger.info("Training samples: {:5d}".format(len(train_datasets)))
    logger.info("Validate samples: {:5d}".format(len(valid_datasets)))
    logger.info("Testing samples: {:5d}".format(len(test_datasets)))

    print("Loaded dataset!")
    print("Training samples: {:5d}".format(len(train_datasets)))
    print("Validate samples: {:5d}".format(len(valid_datasets)))
    print("Testing samples: {:5d}".format(len(test_datasets)))
    """
    Model
    """
    logger.info("Loading model")
    print("Loading model")
    model  = Model(len(genre2idx), use_title=args.use_title).to(device)
    logger.info("Loaded model!")
    print("Loaded model!")

    """
    Loss, Metric, Optimizer
    """
    critical  = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), args.lr)
    
    start_epoch = 1
    if args.check_point is not None:
        logger.info("Load checkpoint " + args.check_point)
        print("Load checkpoint " + args.check_point)
        
        checkpoint = torch.load(args.check_point)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        start_epoch = epoch + 1

        logger.info("Load checkpoint done!")
        logger.info("Start training from " + args.check_point)
        print("Load checkpoint done!")
        print("Start training from " + args.check_point)

    logger.info("Training...........")
    print("Training...........")
    for e in range(start_epoch, args.epoch + 1):
        model.train()
        t_i = time.time()
        for i, (img, title, genre) in enumerate(train_dataloader):

            img = img.to(device)
            if title is not None:
                title = title.to(device)
            genre = genre.to(device)

            out = model(img, title)
            loss = critical(out, genre)

            # img = img.to("cpu")
            # if title is not None:
            #     title = title.to("cpu")
            # genre = genre.to("cpu")

            loss.backward()
            optimizer.zero_grad()
            optimizer.step()

            reduce_Lr(optimizer)

            if i % args.iter_print == 0 and i > 0:
                logger.info("|[TRAIN] epoch : {:5d}| {:5d}/{:5d} batches| time: {:8.2f}s| loss: {:8.3f}|".format(
                    e, i, len(train_dataloader), time.time() - t_i, loss.item()
                ))

                print("|[TRAIN] epoch : {:5d}| {:5d}/{:5d} batches| time: {:8.2f}s| loss: {:8.3f}|".format(
                    e, i, len(train_dataloader), time.time() - t_i, loss.item()
                )) 
                t_i = time.time()

        t_v = time.time()
        p = 0
        r = 0
        f = 0
        l = 0
        for i, (img, title, genre) in enumerate(valid_dataloader):
            model.eval()
            with torch.no_grad():
                img = img.to(device)
                if args.use_title:
                    title = title.to(device)
                genre = genre.to(device)

                out = model(img, title)
                loss = critical(out, genre)
                f1, _p, r = f1_scores(out, genre)
                f += f1
                p += _p
                r += r
                l += loss.item()
        logger.info("|[VALID] epoch : {:5d}| time: {:8.2f}s| loss: {:8.3f}| precission: {:5.3f}| recall: {:5.3f}| f1_score: {:5.3f}|".format(
                    e, time.time() - t_i, l / len(valid_dataloader), p / len(valid_dataloader), r / len(valid_dataloader), f1_scores / len(valid_dataloader) 
                ))
        print("|[VALID] epoch : {:5d}| time: {:8.2f}s| loss: {:8.3f}| precission: {:5.3f}| recall: {:5.3f}| f1_score: {:5.3f}|".format(
                    e, time.time() - t_i, l / len(valid_dataloader), p / len(valid_dataloader), r / len(valid_dataloader), f1_scores / len(valid_dataloader) 
                ))
        save_path = "epoch_" + str(e)
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.save_path, save_path))

if __name__ == "__main__":

    random.seed(1000)
    
    parse = argparse.ArgumentParser()
    parse.add_argument("--lr", type=float, default=0.01, help="Enter learning rate")
    parse.add_argument("--batch_size", type=int, default=32, help="Enter batch size")
    parse.add_argument("--epoch", type=int, default=30, help="Enter epoch")    
    parse.add_argument("--use_title", type=bool, default=False, help="Use title True or False")
    parse.add_argument("--path_data", type=str, help="Enter path_data")
    parse.add_argument("--iter_print", type=int, default=50, help="Enter inter log and print")
    parse.add_argument("--save_path", type=str, default="./models", help="Enter forder to save model")
    parse.add_argument("--check_point", type=str, default=None, help="Enter checkpoint will start")

    args = parse.parse_args()
    
    if not os.path.exists("./logging"):
        os.mkdir("./logging")
    
    now = datetime.now()
    name_logging = "train" + now.strftime("%m_%d_%Y_%H_%M_%S") + ".log"

    path_logging = os.path.join("./logging", name_logging)
    # Create and configure logger
    logging.basicConfig(filename=path_logging,
                    format='%(levelname)s:%(name)s:%(message)s',
                    filemode='w')
    # Creating an object
    logger = logging.getLogger()
    logger.setLevel(logging.BASIC_FORMAT)
    t = time.time()
    train(args, logger)
    logger.info("Total time: {:8.3}s".format(time.time() - t))
    


    
    

    
