import torch


def recall(pred, genre, eps=1e-5):
    sum = 0
    for p, g in zip(pred, genre):
        if p == g and g == 1.0:
            sum += 1
    
    return sum / (genre.sum().item() + eps)

def precision(pred, genre, eps=1e-5):
    sum = 0
    for p, g in zip(pred, genre):
        if p == g and g == 1.0:
            sum += 1
    
    return sum / (pred.sum().item() + eps)

def f1_score(pred, genre, eps=1e-5):
    r = recall(pred, genre)
    p = precision(pred, genre)

    if r == 0 and p == 0:
        return 0, p, r
    
    return 2 * p * r/ (p + r), p, r

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