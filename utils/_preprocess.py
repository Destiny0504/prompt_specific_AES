import random
import pickle

asap_ranges = {
    1: (2.0, 12.0),
    2: (1.0, 6.0),
    3: (0.0, 3.0),
    4: (0.0, 3.0),
    5: (0.0, 4.0),
    6: (0.0, 4.0),
    7: (0.0, 30.0),
    8: (0.0, 60.0),
    9: (0.5, 9.0),
    10: (1.0, 24.0),
}


asap_essay_lengths = {
    1: 649,
    2: 704,
    3: 219,
    4: 203,
    5: 258,
    6: 289,
    7: 371,
    8: 1077,
    9: 415,
    10: 1024,
    11: 252
}


def fix_score(score, prompt):
    """
    fix the predicted score
    """
    min_score, max_score = asap_ranges[prompt]
    score = score * (max_score - min_score) + min_score
    if score < min_score:
        return min_score
    elif score > max_score:
        return max_score
    else:
        return round(score)

def is_zh(s):
    # '包含汉字的返回TRUE'
    for c in s:
        if c >= '\u4e00' and c <= '\u9fa5':
            return True
    return False


def load_asap_data(data_file, mode, prompt_id):
    texts = list()
    labels = list()
    if mode == 'train':
        with open(f"{data_file}/{prompt_id}_train.pk", 'rb') as f:
            dataset = pickle.load(f)
            texts = [data["content_text"] for data in dataset]
            labels = [int(data["score"]) for data in dataset]
    else:
        with open(f"{data_file}/{prompt_id}_test.pk", 'rb') as f:
            dataset = pickle.load(f)
            texts = [data["content_text"] for data in dataset]
            labels = [int(data["score"]) for data in dataset]

    for text, label in zip(texts, labels):
        yield (text, label)