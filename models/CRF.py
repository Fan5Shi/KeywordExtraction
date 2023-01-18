import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import time
import pandas as pd
from label2id import *
import pickle

import pycrfsuite

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
stops = set(stopwords.words('english'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

seed = 0
torch.manual_seed(seed)


def datasetForClassification(data2):
    Index, SentNum, Word, POS, Tag = [],[],[],[],[]
    for i in range(len(data2)):
        tokens = data2['tokens'].iloc[i]
        labels = data2['labels'].iloc[i]
        sentNum = [i+1] * len(tokens)
        index = list(range(len(tokens)))
        pos = [b for a, b in pos_tag(tokens)]

        Index.extend(index)
        SentNum.extend(sentNum)
        Word.extend(tokens)
        POS.extend(pos)
        Tag.extend(labels)

    return pd.DataFrame({"Index": Index, "Sentence #": SentNum,
                            "Word": Word, "POS": POS, "Tag": Tag})

# A class to retrieve the sentences from the dataset
class getsentence(object):
    
    def __init__(self, data):
        self.n_sent = 1.0
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

# Feature set
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
        
    return features

def tags_to_keywords_crf(sample, words):
    indices = [i for i, l in enumerate(sample) if l != 'O']
    keywords, key_cats = [], []
    for j, id in enumerate(indices):
        if j == 0:
            start = end = id
            continue
        if j == len(indices):
            pos = words[start:end+1]
            if (pos[-1]['postag'] == 'NN' or pos[-1]['postag'] == 'NNS' or pos[-1]['postag'] == 'NNP' or pos[-1]['postag'] == 'NNPS') and pos[0]['postag'] != 'CC':
                keywords.append(' '.join([p['word.lower()'] for p in pos]))
                key_cats.append((' '.join([p['word.lower()'] for p in pos]), sample[start:end+1]))
            continue
        if end+1 == id:
            end = id
        else:
            pos = words[start:end+1]
            if (pos[-1]['postag'] == 'NN' or pos[-1]['postag'] == 'NNS' or pos[-1]['postag'] == 'NNP' or pos[-1]['postag'] == 'NNPS') and pos[0]['postag'] != 'CC':
                keywords.append(' '.join([p['word.lower()'] for p in pos]))
                key_cats.append((' '.join([p['word.lower()'] for p in pos]), sample[start:end+1]))
            start = end = id
    return list(set(keywords)), key_cats

directory_file = '/Users/revekkakyriakoglou/Documents/paris_8/Projects/KeywordExtraction/'
data_file = directory_file + 'data/finalData/'
save_file = directory_file + 'models/results/'

model_name = "CRF"

def tokenize_example(example, max_length=300):
    return example.split(' ')[:max_length]

def split_tags(example, max_length=300):
    return example.split(',')[:max_length]

for trainingSession in range(1, 5):

    with open(data_file + f"trainset-{trainingSession}.pkl", "rb") as f:
        train_set = pickle.load(f)
    with open(data_file + f"testset-{trainingSession}.pkl", "rb") as f:
        test_set = pickle.load(f)

    train_set['tokens'] = train_set['sentence'].map(tokenize_example)
    train_set['labels'] = train_set['word_labels'].map(split_tags)
    test_set['tokens'] = test_set['sentence'].map(tokenize_example)
    test_set['labels'] = test_set['word_labels'].map(split_tags)

    train_ml = datasetForClassification(train_set)
    test_ml = datasetForClassification(test_set)

    train_getter = getsentence(train_ml)
    test_getter = getsentence(test_ml)
    train_sentences = train_getter.sentences
    test_sentences = test_getter.sentences

    X_train = [sent2features(s) for s in train_sentences]
    y_train = [sent2labels(s) for s in train_sentences]
    X_test = [sent2features(s) for s in test_sentences]
    y_test = [sent2labels(s) for s in test_sentences]

    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    
    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    print(f"{model_name} start:")
    trainer.train(save_file + 'conll2002-esp.crfsuite')
    tagger = pycrfsuite.Tagger()
    tagger.open(save_file + 'conll2002-esp.crfsuite')

    y_pred = [tagger.tag(xseq) for xseq in X_test]  

    print("Predicting...")
    Preds, Preds_cats, Lbs, Lbs_cats = [], [], [], []
    for n, pred in enumerate(y_pred):
        sentence = X_test[n]
        preds, preds_cats = tags_to_keywords_crf(pred, sentence)
        lbs, lbs_cats = tags_to_keywords_crf(y_test[n], sentence)

        Preds.append(preds)
        Preds_cats.append(preds_cats)
        Lbs.append(lbs)
        Lbs_cats.append(lbs_cats)
        
    pred_lable_keywords = {}
    pred_lable_keywords['predicitons'] = Preds
    pred_lable_keywords['preds_cats'] = Preds_cats
    pred_lable_keywords['labels'] = Lbs
    pred_lable_keywords['labels_cats'] = Lbs_cats

    print("Storing the model ...")
    with open(save_file + f"./{model_name}-label-pred-keywords-{trainingSession}.pkl", "wb") as f:
        pickle.dump(pred_lable_keywords, f)