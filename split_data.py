import argparse
import torch
import numpy
import random
import os

# from src.models.retnet import Retnet
from datasets import get_data

if __name__ == "__main__":
    parse = argparse.ArgumentParser()  
    parse.add_argument("--path_data", type=str, help="Enter path_data")

    args = parse.parse_args()
        
    movies_train = get_data(args.path_data)
    movies_valid = movies_train.sample(frac=0.2)
    movies_train = movies_train.drop(movies_valid.index)
    movies_test = get_data(args.path_data, mode="test")

    movies_train.to_csv(os.path.join(args.path_data, "movies_train.csv"), index=False)
    movies_valid.to_csv(os.path.join(args.path_data, "movies_valid.csv"), index=False)
    movies_test.to_csv(os.path.join(args.path_data, "movies_test.csv"), index=False)

    print("Train: ", len(movies_train))
    print("Valid: ", len(movies_valid))
    print("Test: ", len(movies_test))