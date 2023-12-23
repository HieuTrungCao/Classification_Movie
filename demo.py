import argparse
import os
import torch
import cv2
from transformers import AutoTokenizer

# from src.models.retnet import Retnet
from models import ModelWithBert

def test(args):
    device = "cpu" 
    if torch.cuda.is_available():
        device = "cuda"

    with open(os.path.join(args.path_data, 'genres.txt'), 'r') as f:
        genre_all = f.readlines()
        genre_all = [x.replace('\n','') for x in genre_all]
    
    genre2idx = {genre:idx for idx, genre in enumerate(genre_all)}

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = ModelWithBert(num_classes=len(genre_all), title_model=args.title_model)
    # model.to(device)
    i = args.model.find("_", -5, -1)
    m = "epoch" + args.model[ i: ]
    checkpoint = torch.load(os.path.join(args.model, m))
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    
    model.eval()
    
    img = cv2.imread(args.path_image)
             
    img = cv2.resize(img, (224, 224))
    img_tensor = torch.from_numpy(img.transpose(2,0,1)).float()
    img_tensor = torch.unsqueeze(img_tensor, 0).to(device)

    title_tensor = tokenizer.encode_plus(
            args.title,
            add_special_tokens=True,
            max_length= 40,
            return_token_type_ids=True,
            padding='max_length',
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
    
    title_tensor["input_ids"] = title_tensor['input_ids'][:, 0, :].to(device)
    title_tensor["attention_mask"] = title_tensor['attention_mask'][:, 0, :].to(device)
    title_tensor["token_type_ids"] = title_tensor['token_type_ids'][:, 0, :].to(device)
    
    out = model(img_tensor, title_tensor)
    
    out = torch.sigmoid(out)
    out = out[0]

    label = {}
    for i, o in enumerate(out):
        if o > args.threshold:
            label[genre_all[i]] = o 

    print(label)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--model", type=str, help="Enter model")
    parse.add_argument("--path_data", type=str, default="./data/dataset", help="Enter path data to load genre")
    parse.add_argument("--path_image", type=str, default="./data/dataset", help="Enter path_data")
    parse.add_argument("--title", type=str, default="", help="Enter title image")
    parse.add_argument("--threshold", type=float, default=0.6, help="Enter threshold")
    parse.add_argument("--save_result", type=str, default="/content/result.csv")
    parse.add_argument("--title_model", type=str, default="distilbert-base-uncased")
    
    args = parse.parse_args()

    test(args)