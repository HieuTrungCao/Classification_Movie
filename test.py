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
import json

from datetime import datetime
from torch.utils.data import DataLoader
from torch import optim
import torchvision.transforms as transforms
from torchmetrics.classification import MultilabelF1Score, MultilabelRecall, MultilabelPrecision

# from src.models.retnet import Retnet
from models import Model
from datasets import get_dataframe
from datasets import MyDataset
# from metrics import f1_scores
from utils import print_log
from datasets import Vocab

def get_preds(preds, genres, outs, genre, threshold, genre_all):
    for o, g in zip(outs, genre):
        p = ""
        for i, v in enumerate(o):
            if v >= threshold:
                p = p + genre_all[i] + "|"
        preds.append(p)

        g_s = ""
        for i, v in enumerate(g):
            if v >= threshold:
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

    config = json.load(open(os.path.join(args.model, "config.json")))
    vocabs = json.load(open(os.path.join(args.model, "vocab.json")))
    vocab = Vocab(config["max_length"], get_year=config["get_year"], vocab=vocabs)

    movies_test = get_dataframe(os.path.join(args.path_data, "movies_test.csv"))
    
    test_datasets = MyDataset(movies_test, genre2idx, get_year=config["get_year"], vocab=vocab)
    test_dataloader = DataLoader(test_datasets, batch_size=args.batch_size)

    model = Model(vocab_size=len(vocabs),
                  embedding_dim=config["embedding_dim"],
                  hidden_dim=config["hidden_state_title"],
                  num_layers=config["num_layers"],
                  num_classes=len(genre_all),
                  pretrained=False)
    # model.to(device)
    i = args.model[-1]
    m = "epoch_" + i
    checkpoint = torch.load(os.path.join(args.model, m))
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    
    model.eval()

    f1_scores = MultilabelF1Score(num_labels=len(genre_all), threshold=args.threshold)
    f1_scores = f1_scores.to(device)
    recall_scores = MultilabelRecall(num_labels=len(genre_all), threshold=args.threshold)
    recall_scores = recall_scores.to(device)
    precision_scores = MultilabelPrecision(num_labels=len(genre_all), threshold=args.threshold)
    precision_scores = precision_scores.to(device)

    f = 0
    p = 0
    r = 0
    t = time.time()

    preds = []
    genres = []

    for i, (img, title, genre) in enumerate(test_dataloader):
        
        img = img.to(device)
        if config["use_title"]:
            title = title.to(device)
        genre = genre.to(device)

        out = model(img, title)
        f1 = f1_scores(out, genre).item()
        _p = precision_scores(out, genre).item()
        _r = recall_scores(out, genre).item()
        f += f1
        p += _p
        r += _r

        get_preds(preds, genres, torch.sigmoid(out), genre, args.threshold, genre_all)

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
    parse.add_argument("--threshold", type=float, default=0.7, help="Enter threshold")
    parse.add_argument("--save_result", type=str, default="/content/result.csv")
    
    args = parse.parse_args()

    test(args)