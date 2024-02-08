def recall_At_k(pred, gt, k=1):
    pred = pred[:k]
    return len(set(pred) & set(gt)) / len(gt)

def precision_At_k(pred, gt, k=1):
    pred = pred[:k]
    return len(set(pred) & set(gt)) / k

def AP(pred, gt, k=1):
    pred = pred[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(pred):
        if p in gt:
            num_hits += 1
            score += num_hits / (i + 1.0)
    return score / min(len(gt), k)

def NDCG(pred, gt, k=1):
    pred = pred[:k]
    dcg = 0.0
    idcg = sum([1.0 / (i + 1) for i in range(min(len(gt), k))])
    for i, p in enumerate(pred):
        if p in gt:
            dcg += 1.0 / (i + 1)
    return dcg / idcg

def MRR(pred, gt, k=1):
    pred = pred[:k]
    for i, p in enumerate(pred):
        if p in gt:
            return 1.0 / (i + 1)
    return 0.0

def MAP(pred, gt, k=1):
    pred = pred[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(pred):
        if p in gt:
            num_hits += 1
            score += num_hits / (i + 1.0)
    return score / len(gt)

def MR(pred, gt):
    rank_list = []
    for i, p in enumerate(pred):
        if p in gt:
            rank_list.append(i + 1)
    
    return sum(rank_list) / len(rank_list)
    