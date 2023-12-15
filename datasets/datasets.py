import torch
import os
import cv2
import numpy as np

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, genre2idx):
        super(MyDataset, self).__init__()
        self.data = data
        self.genre2idx = genre2idx
        self.count = 0
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data.iloc[index].img_path
        genre = self.data.iloc[index].genre

        # preprocess img
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            self.count += 1
        else:
            img = np.random.rand(256,256,3)
        img = cv2.resize(img, (256,256))
        img_tensor = torch.from_numpy(img.transpose(2,0,1)).float()

        # preprocess label
        genre_vector = np.zeros(len(self.genre2idx))

        genre = genre.split("|")
        for g in genre:
            genre_vector[self.genre2idx[g]] = 1
        genre_tensor = torch.from_numpy(genre_vector).float()

        return img_tensor, torch.randn((1, 2, 3)), genre_tensor
 