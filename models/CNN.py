import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import time
from label2id import *
import pickle

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

from torch.nn.utils.rnn import pad_sequence
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.modules import ConditionalRandomField


class CNN_n(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, dropout_rate, pad_index, crf=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.conv1=torch.nn.Conv1d(embedding_dim, 128, 5, padding=2)
        self.conv2=torch.nn.Conv1d(embedding_dim, 128, 3, padding=1)
        self.conv3=torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv4=torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv5=torch.nn.Conv1d(256, 256, 5, padding=2)
        self.fc = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.crf_flag=crf
        if self.crf_flag:
            self.crf = ConditionalRandomField(output_dim) 
        
    def forward(self, ids, ids_len, tag, is_training=True):
        # ids = [batch size, seq len]
#         print(f"ids: {ids.size()}")
        embedded = self.dropout(self.embedding(ids))
#         print(f"embedding: {embedded.size()}")
        # embedded = [batch size, seq len, embedding dim]
        conved = embedded.transpose(1, 2)
#         print(f"after permute: {conved.size()}")
        # embedded = [batch size, embedding dim, seq len]
        x_conv=torch.nn.functional.relu(torch.cat((self.conv1(conved), self.conv2(conved)), dim=1))
#         x_conv=torch.nn.functional.relu(self.conv1(conved))
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv3(x_conv))
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv4(x_conv))
        x_conv=self.dropout(x_conv)
        x_conv=torch.nn.functional.relu(self.conv5(x_conv))
        conved=x_conv.transpose(1, 2)
        logit = self.fc(conved)
        if not is_training:
            if self.crf_flag:
                score = self.crf.viterbi_tags(logit)
            else:
                x_logit = logit.transpose(2, 0)
                score = torch.nn.functional.log_softmax(x_logit).transpose(2, 0)
        else:
            if self.crf_flag:
                score = - self.crf(logit, tag)
            else:
                x_logit = torch.nn.utils.rnn.pack_padded_sequence(logit, ids_len, batch_first=True)
                score = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x_logit.data), tag.data)
        return score

def sent2ids(text):
    return [vocab[w] for w in text]

def labels2ids(labels):
    return [labels_to_ids2[w] for w in labels]

def data_generator(sents, labels, batch_size=32, is_training=True, index=0):
    if is_training:
        select_indices = np.random.choice(len(sents), batch_size, replace=False)
    else:
        start = index
        end = min(start + batch_size, len(sents)) 
        select_indices = list(range(start, end))
    #select_indices = list(range(batch_size))
    batch_sents = np.array(sents)[select_indices]
    batch_labels = np.array(labels)[select_indices]
    
    batch_sents = list(map(sent2ids, batch_sents))
    batch_labels = list(map(labels2ids, batch_labels))
    
    seq_lens = [len(s) for s in batch_sents]
    seq_lens = torch.LongTensor(seq_lens)
    max_len = max(seq_lens)
    
    batch_sents = [torch.LongTensor(s) for s in batch_sents]
    
    batch_sents = pad_sequence(batch_sents, batch_first=True)

    if not is_training:
        return batch_sents, batch_labels, seq_lens, end
    batch_labels = [torch.LongTensor(s) for s in batch_labels]
    batch_labels = pad_sequence(batch_labels, batch_first=True)
  
    return batch_sents, batch_labels, seq_lens

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
            try:
                pos = pos_tag(words[start:end+1])
                if (pos[-1][1] == 'NN' or pos[-1][1] == 'NNS' or pos[-1][1] == 'NNP' or pos[-1][1] == 'NNPS') and pos[0][1] != 'CC':
                    keywords.append(' '.join(words[start:end+1]))
                    key_cats.append((' '.join(words[start:end+1]), sample[start:end+1]))
                start = end = id
            except:
                continue
    return list(set(keywords)), key_cats

directory_file = '/Users/revekkakyriakoglou/Documents/paris_8/Projects/KeywordExtraction/'
data_file = directory_file + 'data/finalData/'
save_file = directory_file + 'models/results/'

min_freq = 3
epoch = 40
embedding_dim = 300
dropout_rate = 0.5
model_name = "CNN-CRF"
special_tokens = ['<unk>', '<pad>']
output_dim = len(labels_to_ids2)

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

    vocab = torchtext.vocab.build_vocab_from_iterator(train_set['tokens'],
                                                    min_freq=min_freq,
                                                    specials=special_tokens)

    unk_index = vocab['<unk>']
    pad_index = vocab['<pad>']
    vocab.set_default_index(unk_index)
    vocab_size = len(vocab)

    model = CNN_n(vocab_size, embedding_dim, output_dim, dropout_rate, pad_index, crf=True)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    model.apply(initialize_weights)

    vectors = torchtext.vocab.GloVe()
    pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())

    model.embedding.weight.data = pretrained_embedding
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    model = model.to(device)

    train_sent_words = train_set['tokens']
    train_sent_tags = train_set['labels']
    loop_num = int(len(train_sent_words)/100)
    print(f"The loop numbers: {loop_num}")
    for i in range(epoch):
        print(f"Epoch {i}:")
        start = time.time()
        model.train()
        for j in range(loop_num):
            optimizer.zero_grad()
            batch_sents, batch_tags, seq_lens, = data_generator(train_sent_words, train_sent_tags, batch_size=100)
            loss = model(batch_sents.to(device), seq_lens.to(device), batch_tags.to(device))
            loss.backward()
            optimizer.step()
            if j % 100 == 0:
                print(f'Loss: {loss.item()} \t Cost time per step: {time.time() - start}')

    dev_sent_words = test_set['tokens']
    dev_sent_tags = test_set['labels']
    model.eval()
    index = 0
    Preds, Preds_cats, Lbs, Lbs_cats = [], [], [], []
    # pbar = tqdm.tqdm(total=len(dev_sent_words))
    while index < len(dev_sent_words):
        batch_sents, batch_tags, seq_lens, index = data_generator(dev_sent_words, 
                                                                dev_sent_tags, batch_size=32, 
                                                                is_training=False, index=index)
        pred_labels = model(batch_sents.to(device), seq_lens.to(device), batch_tags, is_training=False)
        for i, label_seq in enumerate(pred_labels):
            pred_label = [ids_to_labels2[t] for t in label_seq[0]]
            
            preds, preds_cats = tags_to_keywords(pred_label, list(dev_sent_words[index-32:min(index, len(dev_sent_words))])[i])
            lbs, lbs_cats = tags_to_keywords(list(dev_sent_tags[index-32:min(index, len(dev_sent_tags))])[i], 
                                            list(dev_sent_words[index-32:min(index, len(dev_sent_words))])[i])

    #         pred_label_list.append(pred_label)
            Preds.append(preds)
            Preds_cats.append(preds_cats)
            Lbs.append(lbs)
            Lbs_cats.append(lbs_cats)
            
#     pbar.update(1000)
    # pbar.close()

    pred_lable_keywords = {}
    pred_lable_keywords['predicitons'] = Preds
    pred_lable_keywords['preds_cats'] = Preds_cats
    pred_lable_keywords['labels'] = Lbs
    pred_lable_keywords['labels_cats'] = Lbs_cats

    with open(save_file + f"./{model_name}-label-pred-keywords-{trainingSession}.pkl", "wb") as f:
        pickle.dump(pred_lable_keywords, f)

    torch.save(model, save_file + f"./{model_name}-model-{trainingSession}.pt")