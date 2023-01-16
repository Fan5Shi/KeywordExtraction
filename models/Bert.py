import numpy as np
import pandas as pd
import pickle
import time
from label2id import *
from model_training import *

from sklearn.metrics import accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
stops = set(stopwords.words('english'))

import torch
import torchtext
import torch.nn as nn
import torch.nn.init
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from transformers import RobertaForTokenClassification, RobertaTokenizerFast
from transformers import AlbertForTokenClassification, AlbertTokenizerFast

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

seed = 442
np.random.seed(seed)
torch.manual_seed(seed)

class dataset2(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        # step 1: get the sentence and word labels 
        sentence = self.data.sentence[index].strip().split()  
        word_labels = self.data.word_labels[index].split(",") 

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                                    is_split_into_words=True, 
                                    return_offsets_mapping=True, 
                                    padding='max_length', 
                                    truncation=True, 
                                    max_length=self.max_len)
        
        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [labels_to_ids2[label] for label in word_labels] 
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        
        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
            # overwrite label
                encoded_labels[idx] = labels[i]
                i += 1

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        
        return item

    def __len__(self):
        return self.len

# Defining the training function on the 80% of the dataset for tuning the bert model
def train2(model2, optimizer2, epoch, training_loader2):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels, tr_probs = [], [], []
    # put model in training mode
    model2.train()
    
    for idx, batch in enumerate(training_loader2):
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)

        results = model2(input_ids=ids, attention_mask=mask, labels=labels)
        loss = results.loss
        tr_logits = results.logits
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
        
        if idx % 100==0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training loss per 100 training steps: {loss_step}")
           
        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model2.num_labels) # shape (batch_size * seq_len, num_labels)
#         flattened_probabilities = F.softmax(active_logits, dim=1) # probabilities
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
#         probabilities = []
#         for i, act in enumerate(active_accuracy):
#             if act:
#                 probabilities.append(flattened_probabilities[i])
        
        tr_labels.extend(labels)
        tr_preds.extend(predictions)
#         tr_probs.extend(probabilities)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
    
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model2.parameters(), max_norm=MAX_GRAD_NORM
        )
        
        # backward pass
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")

    return epoch_loss, tr_accuracy#, tr_labels, tr_preds, tr_probs

def valid(model, testing_loader):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            
            results = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = results.loss
            eval_logits = results.logits

            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
        
            # if idx % 100==0:
            #     loss_step = eval_loss/nb_eval_steps
            #     print(f"Validation loss per 100 evaluation steps: {loss_step}")
              
            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_probabilities = F.softmax(active_logits, dim=1) # probabilities
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            probabilities = []
            for i, act in enumerate(active_accuracy):
                if act:
                    probabilities.append(flattened_probabilities[i])
            
            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return eval_loss, eval_accuracy

def bert_processing(train_set, test_set, trainingSession):
    training_set = dataset2(train_set, tokenizer, MAX_LEN)
    testing_set = dataset2(test_set, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 4
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': 4
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    trLosslist, trAcclist, evalLosslist, evalAcclist, timelist = [], [], [], [], []
    label_pred_dict = {}
    for epoch in range(EPOCHS):
        print(f"Training epoch: {epoch + 1}")
        tmp_dict = {}
        start = time.time()
        epoch_loss, tr_accuracy = train2(model, optimizer, epoch, training_loader)
        eval_loss, eval_accuracy = valid(model, testing_loader)
        epoch_time = time.time() - start
        trLosslist.append(epoch_loss)
        trAcclist.append(tr_accuracy)
        evalLosslist.append(eval_loss)
        evalAcclist.append(eval_accuracy)
        timelist.append(epoch_time)

    resultdf = pd.DataFrame({"Epoch": list(range(1, EPOCHS+1)),
                            "Train_loss": trLosslist,
                            "Eval_loss": evalLosslist,
                            "Train_Acc": trAcclist,
                            "Eval_Acc": evalAcclist,
                            "Time": timelist})
    
    # K_times
    k_fold = 5
    gap = int(len(test_set) / k_fold)
    indices = [[i*gap, (i+1)*gap if i+1 != k_fold else len(test_set)] 
            for i in range(k_fold)]
    
    pred_lable_keywords = {}
    pred_lable_keywords['predicitons'] = []
    pred_lable_keywords['preds_cats'] = []
    pred_lable_keywords['labels'] = []
    pred_lable_keywords['labels_cats'] = []
    
    print("Testing ...")
    for cur_fold in range(k_fold):
        tmp_dataset = test_set[indices[cur_fold][0]:indices[cur_fold][1]]
        Preds, Preds_cats, Lbs, Lbs_cats = countKeywords(tmp_dataset, model)
        
        pred_lable_keywords['predicitons'].extend(Preds)
        pred_lable_keywords['preds_cats'].extend(Preds_cats)
        pred_lable_keywords['labels'].extend(Lbs)
        pred_lable_keywords['labels_cats'].extend(Lbs_cats)

    with open(save_file + f"{model_name}-label-pred-keywords-{trainingSession}.pkl", "wb") as f:
        pickle.dump(pred_lable_keywords, f)

    resultdf.to_csv(save_file + f"{model_name}-loss-acc-{trainingSession}.csv")
    
    model.save_pretrained(save_file + f"{model_name}-epoch{EPOCHS}-{trainingSession}")

# NEW ONE newly generated words
def lengthOfTokens(pair):
        return len(pair.split())

def tags_to_keywords(sample, words):
    indices = [i for i, l in enumerate(sample) if l != 'O']
    keywords, key_cats = [], []
    for j, id in enumerate(indices):
        if j == 0:
            start = end = id
            continue
        if j == len(indices):
            pos = pos_tag(words[start:end+1])
            if (pos[-1][1] == 'NN' or pos[-1][1] == 'NNS' or pos[-1][1] == 'NNP' or pos[-1][1] == 'NNPS') and pos[0][1] != 'CC':
                keywords.append(' '.join(words[start:end+1]))
                key_cats.append((' '.join(words[start:end+1]), sample[start:end+1]))
            continue
        if end+1 == id:
            end = id
        else:
            pos = pos_tag(words[start:end+1])
            if (pos[-1][1] == 'NN' or pos[-1][1] == 'NNS' or pos[-1][1] == 'NNP' or pos[-1][1] == 'NNPS') and pos[0][1] != 'CC':
                keywords.append(' '.join(words[start:end+1]))
                key_cats.append((' '.join(words[start:end+1]), sample[start:end+1]))
            start = end = id
    return list(set(keywords)), key_cats

def countKeywords(test_dataset, model):
    kws_pairs = []
    Preds, Preds_cats, Lbs, Lbs_cats = [], [], [], []
    for tmp_num in range(len(test_dataset)):
        sentence = test_dataset["sentence"].iloc[tmp_num]

        inputs = tokenizer(sentence.split(),
                            is_split_into_words=True, 
                            return_offsets_mapping=True, 
                            padding='max_length', 
                            truncation=True, 
                            max_length=MAX_LEN,
                            return_tensors="pt")

        # move to gpu
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)
        # forward pass
        outputs = model(ids, attention_mask=mask)
        logits = outputs.logits

        active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

        tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [ids_to_labels2[i] for i in flattened_predictions.cpu().numpy()]
        wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

        prediction = []
        for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
          #only predictions on first word pieces are important
          if mapping[0] == 0 and mapping[1] != 0:
            prediction.append(token_pred[1])
          else:
            continue
        
        # predictions
        preds, preds_cats = tags_to_keywords(prediction, sentence.split())
        lbs, lbs_cats = tags_to_keywords(test_dataset["word_labels"].iloc[tmp_num].split(','), sentence.split())

        Preds.append(preds)
        Preds_cats.append(preds_cats)
        Lbs.append(lbs)
        Lbs_cats.append(lbs_cats)
        
    return Preds, Preds_cats, Lbs, Lbs_cats

MAX_LEN = 300
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 8
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10

directory_file = '/workspaces/KeywordExtraction/'
data_file = directory_file + 'data/finalData/'
save_file = directory_file + 'models/results/'

for trainingSession in range(1, 5):

    with open(data_file + f"trainset-{trainingSession}.pkl", "rb") as f:
        train_set = pickle.load(f)
    with open(data_file + f"testset-{trainingSession}.pkl", "rb") as f:
        test_set = pickle.load(f)

    models_name = ['Bert-base-cased', 'Roberta']
    for model_name in models_name:
        print(model_name)
        if model_name == 'Bert-base-cased':
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
            model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(labels_to_ids2))
        if model_name == 'Albert':
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
            model = AlbertForTokenClassification.from_pretrained('albert-base-v1', num_labels=len(labels_to_ids2))
        if model_name == 'Roberta':
            tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
            model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=len(labels_to_ids2))

        bert_processing(train_set, test_set, trainingSession)