
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
    mrr = 0.0
    for i, p in enumerate(pred):
        if p in gt:
            mrr += 1.0 / (i + 1)
    return mrr

def MAP(pred, gt, k=1):
    pred = pred[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(pred):
        if p in gt:
            num_hits += 1
            score += num_hits / (i + 1.0)
    return score / len(gt)

def MR(pred, gt, k=None):
    rank_list = []
    for i, p in enumerate(pred):
        if p in gt:
            rank_list.append(i + 1)
    
    return sum(rank_list) / len(rank_list)

function_name_dict = {
    'recall@k': recall_At_k,
    'precision@k': precision_At_k,
    'AP': AP,
    'NDCG': NDCG,
    'MRR': MRR,
    'MAP': MAP,
    'MR': MR
}


def run_eval(function_name, pred_list, gt_list, k=10):
    eval_list = []
    if function_name not in function_name_dict:
        raise ValueError(f'function_name should be one of {list(function_name_dict.keys())}, but got function_name={function_name}')
    for pred, gt in zip(pred_list, gt_list):
        eval_list.append(function_name_dict[function_name](pred, [gt.item()], k))
    return eval_list