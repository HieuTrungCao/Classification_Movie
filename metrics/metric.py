import torch


def recall(pred, genre, eps=1e-5):
    sum = 0
    for p, g in zip(pred, genre):
        if p == g and g == 1.0:
            sum += 1
    if sum == 0:
        sum += eps
    
    if genre.sum().item() == 0:
        return eps
    
    return sum / genre.sum().item()

def precision(pred, genre, eps=1e-5):
    sum = 0
    for p, g in zip(pred, genre):
        if p == g and g == 1.0:
            sum += 1
    if sum == 0:
        sum += eps
    
    if pred.sum().item() == 0:
        return eps
    
    return sum / pred.sum().item()

def f1_score(pred, genre, eps=1e-5):
    r = recall(pred, genre)
    p = precision(pred, genre)
    return 2 / (1/p + 1/r), p, r

def f1_scores(preds, genres, threshold=0.7, eps=1e-5):
    preds = preds > threshold

    f1 = 0.0
    ps = 0.0
    rs = 0.0
    for p, g in zip(preds, genres):
        f, _p, r = f1_score(p, g)
        f1 += f 
        ps += _p
        rs += r
    return f1 / preds.size()[0], ps / preds.size()[0], rs / preds.size()[0] 