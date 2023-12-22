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
from torchmetrics.classification import MultilabelF1Score, MultilabelRecall, MultilabelPrecision, MultilabelAccuracy
from transformers import AutoTokenizer

# from src.models.retnet import Retnet
from models import ModelWithBert
from datasets import get_dataframe
from datasets import MultiDataset
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

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    movies_test = get_dataframe(os.path.join(args.path_data, "movies_test.csv"))
    
    test_datasets = MultiDataset(movies_test, genre2idx, tokenizer=tokenizer)
    test_dataloader = DataLoader(test_datasets, batch_size=args.batch_size)

    model = ModelWithBert(num_classes=len(genre_all), title_model=args.title_model)
    # model.to(device)
    i = args.model.find("_", -5, -1)
    m = "epoch" + args.model[ i: ]
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
    accuracy = MultilabelAccuracy(num_labels=len(genre_all), threshold=args.threshold)
    accuracy = accuracy.to(device)

    f = 0
    p = 0
    r = 0
    a = 0
    t = time.time()

    preds = []
    genres = []

    for i, (img, title, genre) in enumerate(test_dataloader):
        
        img = img.to(device)
        title["input_ids"] = title['input_ids'][:, 0, :].to(device)
        title["attention_mask"] = title['attention_mask'][:, 0, :].to(device)
        title["token_type_ids"] = title['token_type_ids'][:, 0, :].to(device)
        genre = genre.to(device)

        out = model(img, title)
        f1 = f1_scores(out, genre).item()
        _p = precision_scores(out, genre).item()
        _r = recall_scores(out, genre).item()
        _a = accuracy(out, genre).item()
        f += f1
        p += _p
        r += _r
        a += _a

        get_preds(preds, genres, torch.sigmoid(out), genre, args.threshold, genre_all)

    print("|[TEST]| time: {:8.2f}s| accuracy: {:5.3f}| precission: {:5.3f}| recall: {:5.3f}| f1_score: {:5.3f}|".format(
                    time.time() - t, a / len(test_dataloader), p / len(test_dataloader), r / len(test_dataloader), f / len(test_dataloader) 
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
    parse.add_argument("--save_result", type=str, default="/content/result.csv")
    parse.add_argument("--title_model", type=str)
    
    args = parse.parse_args()

    test(args)