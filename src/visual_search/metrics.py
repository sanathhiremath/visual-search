import math

def evaluate_ranking(ranked_ids,relevant_ids,k=5):
    if k < 1 or not relevant_ids or len(ranked_ids)!=len(set(ranked_ids)):
        raise ValueError('Unique ranking, nonempty relevance, positive k required')
    relevant = set(relevant_ids)
    hits = [identity in relevant for identity in ranked_ids[:k]]
    dcg = sum(hit / math.log2(rank+2) for rank,hit in enumerate(hits))
    ideal = sum(1/math.log2(rank+2) for rank in range(min(k,len(relevant))))
    first = next((rank+1 for rank,identity in enumerate(ranked_ids) if identity in relevant),None)
    return {'precision_at_5':sum(hits)/k,'recall_at_5':sum(hits)/len(relevant),'ndcg_at_5':dcg/ideal,'mrr':1/first if first else 0.0}

def check_split(catalog,queries):
    if set(item['id'] for item in catalog) & set(item['id'] for item in queries):
        raise ValueError('Query identities leaked into catalog')
    if set(item['sha256'] for item in catalog) & set(item['sha256'] for item in queries):
        raise ValueError('Identical query image bytes leaked into catalog')
