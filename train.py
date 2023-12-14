import argparse
import torch.nn as nn
import numpy as np
import random
import os
import torch
import logging
import time 
import wandb

from datetime import datetime
from torch.utils.data import DataLoader
from torch import optim
import torchvision.transforms as transforms

# from src.models.retnet import Retnet
from models import Model
from datasets import get_dataframe
from datasets import MyDataset
from metrics import f1_scores
from utils import print_log

def reduce_Lr(optimizer, is_reduce=False):
    # print(optimizer.param_groups)
    lr = 0
    for param_group in optimizer.param_groups:
        lr = param_group["lr"]
        if is_reduce:
            param_group['lr'] = param_group['lr'] / 10
    return lr

def count_parameters(model, rg):
    return sum(p.numel() for p in model.parameters() if p.requires_grad == rg)

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
    print_log(logger, "Loading datasets!")
 
    movies_train = get_dataframe(os.path.join(args.path_data, "movies_train.csv"))
    movies_valid = get_dataframe(os.path.join(args.path_data, "movies_valid.csv"))
    movies_test = get_dataframe(os.path.join(args.path_data, "movies_test.csv"))

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    transform = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
    
    train_datasets = MyDataset(movies_train, genre2idx, transform)
    valid_datasets = MyDataset(movies_valid, genre2idx, transform)
    test_datasets = MyDataset(movies_test, genre2idx, transform)

    train_dataloader = DataLoader(train_datasets, batch_size=args.batch_size)
    valid_dataloader = DataLoader(valid_datasets, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_datasets, batch_size=args.batch_size)
    
    print_log(logger, "Loaded dataset!")
    print_log(logger, "Training samples: {:5d}".format(len(train_datasets)))
    print_log(logger, "Validate samples: {:5d}".format(len(valid_datasets)))
    print_log(logger, "Testing samples: {:5d}".format(len(test_datasets)))

    """
    Model
    """
    print_log(logger, "Loading model")
    model  = Model(len(genre2idx), use_title=args.use_title)
    model = model.to(device)
    training_params = count_parameters(model, rg=True)
    Non_trainable_params = count_parameters(model, rg=False)
    total = training_params + Non_trainable_params
    print_log(logger, "Loaded model!")
    print_log(logger, "Trainable: {:15d}".format(training_params))
    print_log(logger, "Non-Trainable: {:15d}".format(Non_trainable_params))
    print_log(logger, "Total: {:15d}".format(total))
    """
    Loss, Metric, Optimizer
    """
    critical  = nn.CrossEntropyLoss()
    optimizer = optimizer = optim.SGD(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=args.lr,
                    )
    
    start_epoch = 1
    if args.check_point is not None:
        print_log(logger, "Load checkpoint " + args.check_point)
        
        checkpoint = torch.load(args.check_point)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        start_epoch = epoch + 1

        print_log(logger, "Load checkpoint done!")
        print_log(logger, "Start training from " + args.check_point)

    print_log(logger, "Training...........")
    for e in range(start_epoch, args.epoch + 1):
        model.train()
        t_i = time.time()
        for i, (img, title, genre) in enumerate(train_dataloader):
            optimizer.zero_grad()
            img = img.to(device)
            if args.use_title:
                title = title.to(device)
            genre = genre.to(device)

            out = model(img, title)
            loss = critical(torch.sigmoid(out), genre)

            loss.backward()
            optimizer.step()

            if i % args.iter_print == 0 and i > 0:
                print_log(logger, "|[TRAIN] epoch : {:5d}| {:5d}/{:5d} batches| time: {:8.2f}s| loss: {:8.3f}|".format(
                    e, i, len(train_dataloader), time.time() - t_i, loss.item()
                ))
                t_i = time.time()
                wandb.log({"train loss": loss})

        lr = reduce_Lr(optimizer, args.is_reduce_lr)
        wandb.log({"Lr": lr})
        
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
                loss = critical(torch.sigmoid(out), genre)
                f1, _p, r = f1_scores(out, genre, args.threshold)
                f += f1
                p += _p
                r += r
                l += loss.item()
        print_log(logger, "|[VALID] epoch : {:5d}| time: {:8.2f}s| loss: {:8.3f}| precission: {:5.3f}| recall: {:5.3f}| f1_score: {:5.3f}|".format(
                    e, time.time() - t_v, l / len(valid_dataloader), p / len(valid_dataloader), r / len(valid_dataloader), f / len(valid_dataloader) 
                ))
        
        wandb.log({"valid loss": l/len(valid_dataloader)})
        wandb.log({"precission": p / len(valid_dataloader)})
        wandb.log({"recal": r / len(valid_dataloader)})
        wandb.log({"f1_score": f / len(valid_dataloader)})
        
        save_path = "epoch_" + str(e)
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.save_path, save_path))

if __name__ == "__main__":
    
    parse = argparse.ArgumentParser()
    parse.add_argument("--lr", type=float, default=0.01, help="Enter learning rate")
    parse.add_argument("--seed", type=int, default=1000, help="Enter seed")
    parse.add_argument("--batch_size", type=int, default=32, help="Enter batch size")
    parse.add_argument("--epoch", type=int, default=30, help="Enter epoch")    
    parse.add_argument("--use_title", type=bool, default=False, help="Use title True or False")
    parse.add_argument("--path_data", type=str, help="Enter path_data")
    parse.add_argument("--iter_print", type=int, default=50, help="Enter inter log and print")
    parse.add_argument("--save_path", type=str, default="./models", help="Enter forder to save model")
    parse.add_argument("--check_point", type=str, default=None, help="Enter checkpoint will start")
    parse.add_argument("--is_reduce_lr", type=bool, default=False, help="Do you want to reduce lr each epoch")
    parse.add_argument("--threshold", type=float, default=0.7, help="Enter thredhold to classification")

    args = parse.parse_args()
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    # 1. Start a W&B Run
    wandb.login()
    run = wandb.init(
        project="classification-movie",
        notes="My first experiment",
        tags=["baseline", "paper1"],
    )
    
    wandb.config = {"epochs": args.epoch, "learning_rate": args.lr, "batch_size": args.batch_size}

    if not os.path.exists("./logging"):
        os.mkdir("./logging")
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    now = datetime.now()
    name_logging = "train" + now.strftime("%m_%d_%Y_%H_%M_%S") + ".log"

    path_logging = os.path.join("./logging", name_logging)
    # Create and configure logger
    logging.basicConfig(filename=path_logging,
                    format='%(message)s',
                    filemode='w')
    # Creating an object
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    t = time.time()
    train(args, logger)
    wandb.finish()
    print_log(logger, "Total time: {:8.3}s".format(time.time() - t))
    


    
    

    
