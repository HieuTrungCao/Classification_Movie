from torchvision.transforms import transforms

import argparse
import torch.nn as nn
import numpy as np
import random
import os
import torch
import logging
import time 
import pandas as pd

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


def get_preds(preds, genres, outs, genre, thredhold, genre_all):
    for o, g in zip(outs, genre):
        p = ""
        for i, v in enumerate(o):
            if v >= thredhold:
                p = p + genre_all[i] + "|"
        preds.append(p)

        g_s = ""
        for i, v in enumerate(g):
            if v >= thredhold:
                g_s = g_s + genre_all[i] + "|"
        genres.append(g_s)

def test(args):
    device = "cpu" 
    if torch.cuda.is_available():
        device = "cuda"

    with open(os.path.join(args.path_data, 'genres.txt'), 'r') as f:
        genre_all = f.readlines()
        genre_all = [x.replace('\n','') for x in genre_all]
    
    genre2idx = {genre:idx for idx, genre in enumerate(genre_all)}

    movies_valid = get_dataframe(os.path.join(args.path_data, "movies_valid.csv"))

    test_datasets = MyDataset(movies_valid, genre2idx)
    test_dataloader = DataLoader(test_datasets, batch_size=args.batch_size)

    model  = Model(len(genre2idx), use_title=args.use_title)
    # model.to(device)
        
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    
    model.eval()
    f = 0
    p = 0
    r = 0
    t = time.time()

    preds = []
    genres = []

    for i, (img, title, genre) in enumerate(test_dataloader):
        
        img = img.to(device)
        if args.use_title:
            title = title.to(device)
        genre = genre.to(device)

        out = model(img, title)
        f1, _p, r = f1_scores(torch.sigmoid(out), genre, args.threshold)
        f += f1
        p += _p
        r += r

        get_preds(preds, genres, torch.sigmoid(out), genre, args.thredhold, genre_all)

    print("|[TEST]| time: {:8.2f}s| precission: {:5.3f}| recall: {:5.3f}| f1_score: {:5.3f}|".format(
                    time.time() - t, p / len(test_dataloader), r / len(test_dataloader), f / len(test_dataloader) 
                ))   
    result = pd.DataFrame({
        "preds": preds,
        "genres": genres
    })    

    result.to_csv(args.save_result, index=False)
    

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--model", type=str, help="Enter model")
    parse.add_argument("--path_data", type=str, default="./data/dataset", help="Enter path_data")
    parse.add_argument("--batch_size", type=int, default=16, help="Enter batch_size")
    parse.add_argument("--thredhold", type=float, default=0.7, help="Enter thredhold")
    parse.add_argument("--save_result", type=str, default="/content/result.csv")
    parse.add_argument("--use_title", type=bool, default=False)
    
    args = parse.parse_args()

    test(args)