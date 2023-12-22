import torch
import os
import cv2
import numpy as np

from torch.utils.data import Dataset

from .vocab import Vocab

class MyDataset(Dataset):
    def __init__(self, data, genre2idx, vocab, get_year = False):
        super(MyDataset, self).__init__()
        self.data = data
        self.genre2idx = genre2idx
        self.count = 0
        self.get_year = get_year
        self.vocab = vocab
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data.iloc[index].img_path
        title = self.data.iloc[index].title
        genre = self.data.iloc[index].genre

        # preprocess img
        img = cv2.imread(img_path)
        self.count += 1
             
        img = cv2.resize(img, (224, 224))
        img_tensor = torch.from_numpy(img.transpose(2,0,1)).float()
        # img_tensor = img_tensor / 255.0
        # preprocess label
        genre_vector = np.zeros(len(self.genre2idx))

        genre = genre.split("|")
        for g in genre:
            genre_vector[self.genre2idx[g]] = 1
        genre_tensor = torch.from_numpy(genre_vector).float()

        title_tensor = torch.tensor(self.vocab.word_encode(title), dtype=torch.int)

        return img_tensor, title_tensor, genre_tensor
 
class MultiDataset(Dataset):
    def __init__(self, data, genre2idx, tokenizer):
        super(MultiDataset, self).__init__()
        self.data = data
        self.genre2idx = genre2idx
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data.iloc[index].img_path
        title = self.data.iloc[index].title
        genre = self.data.iloc[index].genre

        # preprocess img
        img = cv2.imread(img_path)
        self.count += 1
             
        img = cv2.resize(img, (224, 224))
        img_tensor = torch.from_numpy(img.transpose(2,0,1)).float()
        # img_tensor = img_tensor / 255.0
        # preprocess label
        genre_vector = np.zeros(len(self.genre2idx))

        genre = genre.split("|")
        for g in genre:
            genre_vector[self.genre2idx[g]] = 1
        genre_tensor = torch.from_numpy(genre_vector).float()

        title_tensor = self.tokenizer.encode_plus(
            title,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )

        return img_tensor, title_tensor, genre_tensor 