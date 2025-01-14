# !pip install eli5
# !pip install allennlp
import numpy as np
import pandas as pd
import pickle
import time
import codecs
from collections import Counter
from functools import reduce

from label2id import *
# from model_training import *

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
import scipy.stats
import eli5
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier
# from sklearn_crfsuite import CRF, scorers, metrics
# from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from allennlp.modules import ConditionalRandomField

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

seed = 442
np.random.seed(seed)
torch.manual_seed(seed)

class Highway(nn.Module):
    """
    Highway Network.
    """

    def __init__(self, size, num_layers=1, dropout=0.5):
        """
        :param size: size of linear layer (matches input size)
        :param num_layers: number of transform and gate layers
        :param dropout: dropout
        """
        super(Highway, self).__init__()
        self.size = size
        self.num_layers = num_layers
        self.transform = nn.ModuleList()  # list of transform layers
        self.gate = nn.ModuleList()  # list of gate layers
        self.dropout = nn.Dropout(p=dropout)

        for i in range(num_layers):
            transform = nn.Linear(size, size)
            gate = nn.Linear(size, size)
            self.transform.append(transform)
            self.gate.append(gate)

    def forward(self, x):
        """
        Forward propagation.

        :param x: input tensor
        :return: output tensor, with same dimensions as input tensor
        """
        transformed = nn.functional.relu(self.transform[0](x))  # transform input
        g = torch.sigmoid(self.gate[0](x))  # calculate how much of the transformed input to keep

        out = g * transformed + (1 - g) * x  # combine input and transformed input in this ratio

        # If there are additional layers
        for i in range(1, self.num_layers):
            out = self.dropout(out)
            transformed = nn.functional.relu(self.transform[i](out))
            g = torch.sigmoid(self.gate[i](out))

            out = g * transformed + (1 - g) * out

        return out


class CRF(nn.Module):
    """
    Conditional Random Field.
    """

    def __init__(self, hidden_dim, tagset_size):
        """
        :param hidden_dim: size of word RNN/BLSTM's output
        :param tagset_size: number of tags
        """
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.emission = nn.Linear(hidden_dim, self.tagset_size)
        self.transition = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size))
        self.transition.data.zero_()

    def forward(self, feats):
        """
        Forward propagation.

        :param feats: output of word RNN/BLSTM, a tensor of dimensions (batch_size, timesteps, hidden_dim)
        :return: CRF scores, a tensor of dimensions (batch_size, timesteps, tagset_size, tagset_size)
        """
        self.batch_size = feats.size(0)
        self.timesteps = feats.size(1)

        emission_scores = self.emission(feats)  # (batch_size, timesteps, tagset_size)
        emission_scores = emission_scores.unsqueeze(2).expand(self.batch_size, self.timesteps, self.tagset_size,
                                                              self.tagset_size)  # (batch_size, timesteps, tagset_size, tagset_size)

        crf_scores = emission_scores + self.transition.unsqueeze(0).unsqueeze(
            0)  # (batch_size, timesteps, tagset_size, tagset_size)
        return crf_scores


class LM_LSTM_CRF(nn.Module):
    """
    The encompassing LM-LSTM-CRF model.
    """

    def __init__(self, tagset_size, charset_size, char_emb_dim, char_rnn_dim, char_rnn_layers, vocab_size,
                 lm_vocab_size, word_emb_dim, word_rnn_dim, word_rnn_layers, dropout, highway_layers=1):
        """
        :param tagset_size: number of tags
        :param charset_size: size of character vocabulary
        :param char_emb_dim: size of character embeddings
        :param char_rnn_dim: size of character RNNs/LSTMs
        :param char_rnn_layers: number of layers in character RNNs/LSTMs
        :param vocab_size: input vocabulary size
        :param lm_vocab_size: vocabulary size of language models (in-corpus words subject to word frequency threshold)
        :param word_emb_dim: size of word embeddings
        :param word_rnn_dim: size of word RNN/BLSTM
        :param word_rnn_layers:  number of layers in word RNNs/LSTMs
        :param dropout: dropout
        :param highway_layers: number of transform and gate layers
        """

        super(LM_LSTM_CRF, self).__init__()

        self.tagset_size = tagset_size  # this is the size of the output vocab of the tagging model

        self.charset_size = charset_size
        self.char_emb_dim = char_emb_dim
        self.char_rnn_dim = char_rnn_dim
        self.char_rnn_layers = char_rnn_layers

        self.wordset_size = vocab_size  # this is the size of the input vocab (embedding layer) of the tagging model
        self.lm_vocab_size = lm_vocab_size  # this is the size of the output vocab of the language model
        self.word_emb_dim = word_emb_dim
        self.word_rnn_dim = word_rnn_dim
        self.word_rnn_layers = word_rnn_layers

        self.highway_layers = highway_layers

        self.dropout = nn.Dropout(p=dropout)

        self.char_embeds = nn.Embedding(self.charset_size, self.char_emb_dim)  # character embedding layer
        self.forw_char_lstm = nn.LSTM(self.char_emb_dim, self.char_rnn_dim, num_layers=self.char_rnn_layers,
                                      bidirectional=False, dropout=dropout)  # forward character LSTM
        self.back_char_lstm = nn.LSTM(self.char_emb_dim, self.char_rnn_dim, num_layers=self.char_rnn_layers,
                                      bidirectional=False, dropout=dropout)  # backward character LSTM

        self.word_embeds = nn.Embedding(self.wordset_size, self.word_emb_dim)  # word embedding layer
        self.word_blstm = nn.LSTM(self.word_emb_dim + self.char_rnn_dim * 2, self.word_rnn_dim // 2,
                                  num_layers=self.word_rnn_layers, bidirectional=True, dropout=dropout)  # word BLSTM

        self.crf = CRF((self.word_rnn_dim // 2) * 2, self.tagset_size)  # conditional random field

        self.forw_lm_hw = Highway(self.char_rnn_dim, num_layers=self.highway_layers,
                                  dropout=dropout)  # highway to transform forward char LSTM output for the forward language model
        self.back_lm_hw = Highway(self.char_rnn_dim, num_layers=self.highway_layers,
                                  dropout=dropout)  # highway to transform backward char LSTM output for the backward language model
        self.subword_hw = Highway(2 * self.char_rnn_dim, num_layers=self.highway_layers,
                                  dropout=dropout)  # highway to transform combined forward and backward char LSTM outputs for use in the word BLSTM

        self.forw_lm_out = nn.Linear(self.char_rnn_dim,
                                     self.lm_vocab_size)  # linear layer to find vocabulary scores for the forward language model
        self.back_lm_out = nn.Linear(self.char_rnn_dim,
                                     self.lm_vocab_size)  # linear layer to find vocabulary scores for the backward language model

    def init_word_embeddings(self, embeddings):
        """
        Initialize embeddings with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.word_embeds.weight = nn.Parameter(embeddings)

    def fine_tune_word_embeddings(self, fine_tune=False):
        """
        Fine-tune embedding layer? (Not fine-tuning only makes sense if using pre-trained embeddings).

        :param fine_tune: Fine-tune?
        """
        for p in self.word_embeds.parameters():
            p.requires_grad = fine_tune

    def forward(self, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, wmaps, tmaps, wmap_lengths, cmap_lengths):
        """
        Forward propagation.

        :param cmaps_f: padded encoded forward character sequences, a tensor of dimensions (batch_size, char_pad_len)
        :param cmaps_b: padded encoded backward character sequences, a tensor of dimensions (batch_size, char_pad_len)
        :param cmarkers_f: padded forward character markers, a tensor of dimensions (batch_size, word_pad_len)
        :param cmarkers_b: padded backward character markers, a tensor of dimensions (batch_size, word_pad_len)
        :param wmaps: padded encoded word sequences, a tensor of dimensions (batch_size, word_pad_len)
        :param tmaps: padded tag sequences, a tensor of dimensions (batch_size, word_pad_len)
        :param wmap_lengths: word sequence lengths, a tensor of dimensions (batch_size)
        :param cmap_lengths: character sequence lengths, a tensor of dimensions (batch_size, word_pad_len)
        """
        self.batch_size = cmaps_f.size(0)
        self.word_pad_len = wmaps.size(1)

        # Sort by decreasing true char. sequence length
        cmap_lengths, char_sort_ind = cmap_lengths.sort(dim=0, descending=True)
        cmaps_f = cmaps_f[char_sort_ind]
        cmaps_b = cmaps_b[char_sort_ind]
        cmarkers_f = cmarkers_f[char_sort_ind]
        cmarkers_b = cmarkers_b[char_sort_ind]
        wmaps = wmaps[char_sort_ind]
        tmaps = tmaps[char_sort_ind]
        wmap_lengths = wmap_lengths[char_sort_ind]

        # Embedding look-up for characters
        cf = self.char_embeds(cmaps_f)  # (batch_size, char_pad_len, char_emb_dim)
        cb = self.char_embeds(cmaps_b)

        # Dropout
        cf = self.dropout(cf)  # (batch_size, char_pad_len, char_emb_dim)
        cb = self.dropout(cb)

        # Pack padded sequence
        cf = pack_padded_sequence(cf, cmap_lengths.tolist(),
                                  batch_first=True)  # packed sequence of char_emb_dim, with real sequence lengths
        cb = pack_padded_sequence(cb, cmap_lengths.tolist(), batch_first=True)

        # LSTM
        cf, _ = self.forw_char_lstm(cf)  # packed sequence of char_rnn_dim, with real sequence lengths
        cb, _ = self.back_char_lstm(cb)

        # Unpack packed sequence
        cf, _ = pad_packed_sequence(cf, batch_first=True)  # (batch_size, max_char_len_in_batch, char_rnn_dim)
        cb, _ = pad_packed_sequence(cb, batch_first=True)

        # Sanity check
        assert cf.size(1) == max(cmap_lengths.tolist()) == list(cmap_lengths)[0]

        # Select RNN outputs only at marker points (spaces in the character sequence)
        cmarkers_f = cmarkers_f.unsqueeze(2).expand(self.batch_size, self.word_pad_len, self.char_rnn_dim)
        cmarkers_b = cmarkers_b.unsqueeze(2).expand(self.batch_size, self.word_pad_len, self.char_rnn_dim)
        cf_selected = torch.gather(cf, 1, cmarkers_f)  # (batch_size, word_pad_len, char_rnn_dim)
        cb_selected = torch.gather(cb, 1, cmarkers_b)

        # Only for co-training, not useful for tagging after model is trained
        if self.training:
            lm_f = self.forw_lm_hw(self.dropout(cf_selected))  # (batch_size, word_pad_len, char_rnn_dim)
            lm_b = self.back_lm_hw(self.dropout(cb_selected))
            lm_f_scores = self.forw_lm_out(self.dropout(lm_f))  # (batch_size, word_pad_len, lm_vocab_size)
            lm_b_scores = self.back_lm_out(self.dropout(lm_b))

        # Sort by decreasing true word sequence length
        wmap_lengths, word_sort_ind = wmap_lengths.sort(dim=0, descending=True)
        wmaps = wmaps[word_sort_ind]
        tmaps = tmaps[word_sort_ind]
        cf_selected = cf_selected[word_sort_ind]  # for language model
        cb_selected = cb_selected[word_sort_ind]
        if self.training:
            lm_f_scores = lm_f_scores[word_sort_ind]
            lm_b_scores = lm_b_scores[word_sort_ind]

        # Embedding look-up for words
        w = self.word_embeds(wmaps)  # (batch_size, word_pad_len, word_emb_dim)
        w = self.dropout(w)

        # Sub-word information at each word
        subword = self.subword_hw(self.dropout(
            torch.cat((cf_selected, cb_selected), dim=2)))  # (batch_size, word_pad_len, 2 * char_rnn_dim)
        subword = self.dropout(subword)

        # Concatenate word embeddings and sub-word features
        w = torch.cat((w, subword), dim=2)  # (batch_size, word_pad_len, word_emb_dim + 2 * char_rnn_dim)

        # Pack padded sequence
        w = pack_padded_sequence(w, list(wmap_lengths),
                                 batch_first=True)  # packed sequence of word_emb_dim + 2 * char_rnn_dim, with real sequence lengths

        # LSTM
        w, _ = self.word_blstm(w)  # packed sequence of word_rnn_dim, with real sequence lengths

        # Unpack packed sequence
        w, _ = pad_packed_sequence(w, batch_first=True)  # (batch_size, max_word_len_in_batch, word_rnn_dim)
        w = self.dropout(w)

        crf_scores = self.crf(w)  # (batch_size, max_word_len_in_batch, tagset_size, tagset_size)

        if self.training:
            return crf_scores, lm_f_scores, lm_b_scores, wmaps, tmaps, wmap_lengths, word_sort_ind, char_sort_ind
        else:
            return crf_scores, wmaps, tmaps, wmap_lengths, word_sort_ind, char_sort_ind  # sort inds to reorder, if req.


class ViterbiLoss(nn.Module):
    """
    Viterbi Loss.
    """

    def __init__(self, tag_map):
        """
        :param tag_map: tag map
        """
        super(ViterbiLoss, self).__init__()
        self.tagset_size = len(tag_map)
        self.start_tag = tag_map['<start>']
        self.end_tag = tag_map['<end>']

    def forward(self, scores, targets, lengths):
        """
        Forward propagation.

        :param scores: CRF scores
        :param targets: true tags indices in unrolled CRF scores
        :param lengths: word sequence lengths
        :return: viterbi loss
        """

        batch_size = scores.size(0)
        word_pad_len = scores.size(1)

        # Gold score

        targets = targets.unsqueeze(2)
        scores_at_targets = torch.gather(scores.view(batch_size, word_pad_len, -1), 2, targets).squeeze(
            2)  # (batch_size, word_pad_len)

        # Everything is already sorted by lengths
        scores_at_targets = pack_padded_sequence(scores_at_targets, lengths, batch_first=True)
        gold_score = scores_at_targets.data.sum()

        # All paths' scores

        # Create a tensor to hold accumulated sequence scores at each current tag
        scores_upto_t = torch.zeros(batch_size, self.tagset_size).to(device)

        for t in range(max(lengths)):
            batch_size_t = sum([l > t for l in lengths])  # effective batch size (sans pads) at this timestep
            if t == 0:
                scores_upto_t[:batch_size_t] = scores[:batch_size_t, t, self.start_tag, :]  # (batch_size, tagset_size)
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and log-sum-exp
                # Remember, the cur_tag of the previous timestep is the prev_tag of this timestep
                # So, broadcast prev. timestep's cur_tag scores along cur. timestep's cur_tag dimension
                scores_upto_t[:batch_size_t] = log_sum_exp(
                    scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1)  # (batch_size, tagset_size)

        # We only need the final accumulated scores at the <end> tag
        all_paths_scores = scores_upto_t[:, self.end_tag].sum()

        viterbi_loss = all_paths_scores - gold_score
        viterbi_loss = viterbi_loss / batch_size

        return viterbi_loss


def read_words_tags(file, tag_ind, caseless=True):
    """
    Reads raw data in the CoNLL 2003 format and returns word and tag sequences.

    :param file: file with raw data in the CoNLL 2003 format
    :param tag_ind: column index of tag
    :param caseless: lowercase words?
    :return: word, tag sequences
    """
    with codecs.open(file, 'r', 'utf-8') as f:
        lines = f.readlines()
    words = []
    tags = []
    temp_w = []
    temp_t = []
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            feats = line.rstrip('\n').split()
            temp_w.append(feats[0].lower() if caseless else feats[0])
            temp_t.append(feats[tag_ind])
        elif len(temp_w) > 0:
            assert len(temp_w) == len(temp_t)
            words.append(temp_w)
            tags.append(temp_t)
            temp_w = []
            temp_t = []
    # last sentence
    if len(temp_w) > 0:
        assert len(temp_w) == len(temp_t)
        words.append(temp_w)
        tags.append(temp_t)

    # Sanity check
    assert len(words) == len(tags)

    return words, tags


def create_maps(words, tags, min_word_freq=5, min_char_freq=1):
    """
    Creates word, char, tag maps.

    :param words: word sequences
    :param tags: tag sequences
    :param min_word_freq: words that occur fewer times than this threshold are binned as <unk>s
    :param min_char_freq: characters that occur fewer times than this threshold are binned as <unk>s
    :return: word, char, tag maps
    """
    word_freq = Counter()
    char_freq = Counter()
    tag_map = set()
    for w, t in zip(words, tags):
        word_freq.update(w)
        char_freq.update(list(reduce(lambda x, y: list(x) + [' '] + list(y), w)))
        tag_map.update(t)

    word_map = {k: v + 1 for v, k in enumerate([w for w in word_freq.keys() if word_freq[w] > min_word_freq])}
    char_map = {k: v + 1 for v, k in enumerate([c for c in char_freq.keys() if char_freq[c] > min_char_freq])}
    tag_map = {k: v + 1 for v, k in enumerate(tag_map)}

    word_map['<pad>'] = 0
    word_map['<end>'] = len(word_map)
    word_map['<unk>'] = len(word_map)
    char_map['<pad>'] = 0
    char_map['<end>'] = len(char_map)
    char_map['<unk>'] = len(char_map)
    tag_map['<pad>'] = 0
    tag_map['<start>'] = len(tag_map)
    tag_map['<end>'] = len(tag_map)

    return word_map, char_map, tag_map


def create_input_tensors(words, tags, word_map, char_map, tag_map):
    """
    Creates input tensors that will be used to create a PyTorch Dataset.

    :param words: word sequences
    :param tags: tag sequences
    :param word_map: word map
    :param char_map: character map
    :param tag_map: tag map
    :return: padded encoded words, padded encoded forward chars, padded encoded backward chars,
            padded forward character markers, padded backward character markers, padded encoded tags,
            word sequence lengths, char sequence lengths
    """
    # Encode sentences into word maps with <end> at the end
    # [['dunston', 'checks', 'in', '<end>']] -> [[4670, 4670, 185, 4669]]
    wmaps = list(map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [word_map['<end>']], words))

    # Forward and backward character streams
    # [['d', 'u', 'n', 's', 't', 'o', 'n', ' ', 'c', 'h', 'e', 'c', 'k', 's', ' ', 'i', 'n', ' ']]
    chars_f = list(map(lambda s: list(reduce(lambda x, y: list(x) + [' '] + list(y), s)) + [' '], words))
    # [['n', 'i', ' ', 's', 'k', 'c', 'e', 'h', 'c', ' ', 'n', 'o', 't', 's', 'n', 'u', 'd', ' ']]
    chars_b = list(
        map(lambda s: list(reversed([' '] + list(reduce(lambda x, y: list(x) + [' '] + list(y), s)))), words))

    # Encode streams into forward and backward character maps with <end> at the end
    # [[29, 2, 12, 8, 7, 14, 12, 3, 6, 18, 1, 6, 21, 8, 3, 17, 12, 3, 60]]
    cmaps_f = list(
        map(lambda s: list(map(lambda c: char_map.get(c, char_map['<unk>']), s)) + [char_map['<end>']], chars_f))
    # [[12, 17, 3, 8, 21, 6, 1, 18, 6, 3, 12, 14, 7, 8, 12, 2, 29, 3, 60]]
    cmaps_b = list(
        map(lambda s: list(map(lambda c: char_map.get(c, char_map['<unk>']), s)) + [char_map['<end>']], chars_b))

    # Positions of spaces and <end> character
    # Words are predicted or encoded at these places in the language and tagging models respectively
    # [[7, 14, 17, 18]] are points after '...dunston', '...checks', '...in', '...<end>' respectively
    cmarkers_f = list(map(lambda s: [ind for ind in range(len(s)) if s[ind] == char_map[' ']] + [len(s) - 1], cmaps_f))
    # Reverse the markers for the backward stream before adding <end>, so the words of the f and b markers coincide
    # i.e., [[17, 9, 2, 18]] are points after '...notsnud', '...skcehc', '...ni', '...<end>' respectively
    cmarkers_b = list(
        map(lambda s: list(reversed([ind for ind in range(len(s)) if s[ind] == char_map[' ']])) + [len(s) - 1],
            cmaps_b))

    # Encode tags into tag maps with <end> at the end
    tmaps = list(map(lambda s: list(map(lambda t: tag_map[t], s)) + [tag_map['<end>']], tags))
    # Since we're using CRF scores of size (prev_tags, cur_tags), find indices of target sequence in the unrolled scores
    # This will be row_index (i.e. prev_tag) * n_columns (i.e. tagset_size) + column_index (i.e. cur_tag)
    tmaps = list(map(lambda s: [tag_map['<start>'] * len(tag_map) + s[0]] + [s[i - 1] * len(tag_map) + s[i] for i in
                                                                             range(1, len(s))], tmaps))
    # Note - the actual tag indices can be recovered with tmaps % len(tag_map)

    # Pad, because need fixed length to be passed around by DataLoaders and other layers
    word_pad_len = max(list(map(lambda s: len(s), wmaps)))
    char_pad_len = max(list(map(lambda s: len(s), cmaps_f)))

    # Sanity check
    assert word_pad_len == max(list(map(lambda s: len(s), tmaps)))

    padded_wmaps = []
    padded_cmaps_f = []
    padded_cmaps_b = []
    padded_cmarkers_f = []
    padded_cmarkers_b = []
    padded_tmaps = []
    wmap_lengths = []
    cmap_lengths = []

    for w, cf, cb, cmf, cmb, t in zip(wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps):
        # Sanity  checks
        assert len(w) == len(cmf) == len(cmb) == len(t)
        assert len(cmaps_f) == len(cmaps_b)

        # Pad
        # A note -  it doesn't really matter what we pad with, as long as it's a valid index
        # i.e., we'll extract output at those pad points (to extract equal lengths), but never use them

        padded_wmaps.append(w + [word_map['<pad>']] * (word_pad_len - len(w)))
        padded_cmaps_f.append(cf + [char_map['<pad>']] * (char_pad_len - len(cf)))
        padded_cmaps_b.append(cb + [char_map['<pad>']] * (char_pad_len - len(cb)))

        # 0 is always a valid index to pad markers with (-1 is too but torch.gather has some issues with it)
        padded_cmarkers_f.append(cmf + [0] * (word_pad_len - len(w)))
        padded_cmarkers_b.append(cmb + [0] * (word_pad_len - len(w)))

        padded_tmaps.append(t + [tag_map['<pad>']] * (word_pad_len - len(t)))

        wmap_lengths.append(len(w))
        cmap_lengths.append(len(cf))

        # Sanity check
        assert len(padded_wmaps[-1]) == len(padded_tmaps[-1]) == len(padded_cmarkers_f[-1]) == len(
            padded_cmarkers_b[-1]) == word_pad_len
        assert len(padded_cmaps_f[-1]) == len(padded_cmaps_b[-1]) == char_pad_len

    padded_wmaps = torch.LongTensor(padded_wmaps)
    padded_cmaps_f = torch.LongTensor(padded_cmaps_f)
    padded_cmaps_b = torch.LongTensor(padded_cmaps_b)
    padded_cmarkers_f = torch.LongTensor(padded_cmarkers_f)
    padded_cmarkers_b = torch.LongTensor(padded_cmarkers_b)
    padded_tmaps = torch.LongTensor(padded_tmaps)
    wmap_lengths = torch.LongTensor(wmap_lengths)
    cmap_lengths = torch.LongTensor(cmap_lengths)

    return padded_wmaps, padded_cmaps_f, padded_cmaps_b, padded_cmarkers_f, padded_cmarkers_b, padded_tmaps, \
           wmap_lengths, cmap_lengths


def init_embedding(input_embedding):
    """
    Initialize embedding tensor with values from the uniform distribution.

    :param input_embedding: embedding tensor
    :return:
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def load_embeddings(emb_file, word_map, expand_vocab=True):
    """
    Load pre-trained embeddings for words in the word map.

    :param emb_file: file with pre-trained embeddings (in the GloVe format)
    :param word_map: word map
    :param expand_vocab: expand vocabulary of word map to vocabulary of pre-trained embeddings?
    :return: embeddings for words in word map, (possibly expanded) word map,
            number of words in word map that are in-corpus (subject to word frequency threshold)
    """
    with open(emb_file, 'r') as f:
        emb_len = len(f.readline().split(' ')) - 1

    print("Embedding length is %d." % emb_len)

    # Create tensor to hold embeddings for words that are in-corpus
    ic_embs = torch.FloatTensor(len(word_map), emb_len)
    init_embedding(ic_embs)

    if expand_vocab:
        print("You have elected to include embeddings that are out-of-corpus.")
        ooc_words = []
        ooc_embs = []
    else:
        print("You have elected NOT to include embeddings that are out-of-corpus.")

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        if not expand_vocab and emb_word not in word_map:
            continue

        # If word is in train_vocab, store at the correct index (as in the word_map)
        if emb_word in word_map:
            ic_embs[word_map[emb_word]] = torch.FloatTensor(embedding)

        # If word is in dev or test vocab, store it and its embedding into lists
        elif expand_vocab:
            ooc_words.append(emb_word)
            ooc_embs.append(embedding)

    lm_vocab_size = len(word_map)  # keep track of lang. model's output vocab size (no out-of-corpus words)

    if expand_vocab:
        print("'word_map' is being updated accordingly.")
        for word in ooc_words:
            word_map[word] = len(word_map)
        ooc_embs = torch.FloatTensor(np.asarray(ooc_embs))
        embeddings = torch.cat([ic_embs, ooc_embs], 0)

    else:
        embeddings = ic_embs

    # Sanity check
    assert embeddings.size(0) == len(word_map)

    print("\nDone.\n Embedding vocabulary: %d\n Language Model vocabulary: %d.\n" % (len(word_map), lm_vocab_size))

    return embeddings, word_map, lm_vocab_size


def clip_gradient(optimizer, grad_clip):
    """
    Clip gradients computed during backpropagation to prevent gradient explosion.

    :param optimizer: optimized with the gradients to be clipped
    :param grad_clip: gradient clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, model, optimizer, val_f1, word_map, char_map, tag_map, lm_vocab_size, is_best):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimized
    :param val_f1: validation F1 score
    :param word_map: word map
    :param char_map: char map
    :param tag_map: tag map
    :param lm_vocab_size: number of words in-corpus, i.e. size of output vocabulary of linear model
    :param is_best: is this checkpoint the best so far?
    :return:
    """
    state = {'epoch': epoch,
             'f1': val_f1,
             'model': model,
             'optimizer': optimizer,
             'word_map': word_map,
             'tag_map': tag_map,
             'char_map': char_map,
             'lm_vocab_size': lm_vocab_size}
    filename = 'checkpoint_lm_lstm_crf.pth.tar'
    torch.save(state, filename)
    # If checkpoint is the best so far, create a copy to avoid being overwritten by a subsequent worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, new_lr):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rates must be decayed
    :param new_lr: new learning rate
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def log_sum_exp(tensor, dim):
    """
    Calculates the log-sum-exponent of a tensor's dimension in a numerically stable way.

    :param tensor: tensor
    :param dim: dimension to calculate log-sum-exp of
    :return: log-sum-exp
    """
    m, _ = torch.max(tensor, dim)
    m_expanded = m.unsqueeze(dim).expand_as(tensor)
    return m + torch.log(torch.sum(torch.exp(tensor - m_expanded), dim))

class ViterbiDecoder():
    """
    Viterbi Decoder.
    """

    def __init__(self, tag_map):
        """
        :param tag_map: tag map
        """
        self.tagset_size = len(tag_map)
        self.start_tag = tag_map['<start>']
        self.end_tag = tag_map['<end>']

    def decode(self, scores, lengths):
        """
        :param scores: CRF scores
        :param lengths: word sequence lengths
        :return: decoded sequences
        """
        batch_size = scores.size(0)
        word_pad_len = scores.size(1)

        # Create a tensor to hold accumulated sequence scores at each current tag
        scores_upto_t = torch.zeros(batch_size, self.tagset_size)

        # Create a tensor to hold back-pointers
        # i.e., indices of the previous_tag that corresponds to maximum accumulated score at current tag
        # Let pads be the <end> tag index, since that was the last tag in the decoded sequence
        backpointers = torch.ones((batch_size, max(lengths), self.tagset_size), dtype=torch.long) * self.end_tag

        for t in range(max(lengths)):
            batch_size_t = sum([l > t for l in lengths])  # effective batch size (sans pads) at this timestep
            if t == 0:
                scores_upto_t[:batch_size_t] = scores[:batch_size_t, t, self.start_tag, :]  # (batch_size, tagset_size)
                backpointers[:batch_size_t, t, :] = torch.ones((batch_size_t, self.tagset_size),
                                                               dtype=torch.long) * self.start_tag
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and
                # choose the previous timestep that corresponds to the max. accumulated score for each current timestep
                scores_upto_t[:batch_size_t], backpointers[:batch_size_t, t, :] = torch.max(
                    scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1)  # (batch_size, tagset_size)

        # Decode/trace best path backwards
        decoded = torch.zeros((batch_size, backpointers.size(1)), dtype=torch.long)
        pointer = torch.ones((batch_size, 1),
                             dtype=torch.long) * self.end_tag  # the pointers at the ends are all <end> tags

        for t in list(reversed(range(backpointers.size(1)))):
            decoded[:, t] = torch.gather(backpointers[:, t, :], 1, pointer).squeeze(1)
            pointer = decoded[:, t].unsqueeze(1)  # (batch_size, 1)

        # Sanity check
        assert torch.equal(decoded[:, 0], torch.ones((batch_size), dtype=torch.long) * self.start_tag)

        # Remove the <starts> at the beginning, and append with <ends> (to compare to targets, if any)
        decoded = torch.cat([decoded[:, 1:], torch.ones((batch_size, 1), dtype=torch.long) * self.start_tag],
                            dim=1)

        return decoded

class WCDataset(Dataset):
    """
    PyTorch Dataset for the LM-LSTM-CRF model. To be used by a PyTorch DataLoader to feed batches to the model.
    """

    def __init__(self, wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths):
        """
        :param wmaps: padded encoded word sequences
        :param cmaps_f: padded encoded forward character sequences
        :param cmaps_b: padded encoded backward character sequences
        :param cmarkers_f: padded forward character markers
        :param cmarkers_b: padded backward character markers
        :param tmaps: padded encoded tag sequences (indices in unrolled CRF scores)
        :param wmap_lengths: word sequence lengths
        :param cmap_lengths: character sequence lengths
        """
        self.wmaps = wmaps
        self.cmaps_f = cmaps_f
        self.cmaps_b = cmaps_b
        self.cmarkers_f = cmarkers_f
        self.cmarkers_b = cmarkers_b
        self.tmaps = tmaps
        self.wmap_lengths = wmap_lengths
        self.cmap_lengths = cmap_lengths

        self.data_size = self.wmaps.size(0)

    def __getitem__(self, i):
        return self.wmaps[i], self.cmaps_f[i], self.cmaps_b[i], self.cmarkers_f[i], self.cmarkers_b[i], self.tmaps[i], \
               self.wmap_lengths[i], self.cmap_lengths[i]

    def __len__(self):
        return self.data_size


def train(train_loader, model, lm_criterion, crf_criterion, optimizer, epoch, vb_decoder, word_map, tag_map):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param lm_criterion: cross entropy loss layer
    :param crf_criterion: viterbi loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    :param vb_decoder: viterbi decoder (to decode and find F1 score)
    """

    model.train()  # training mode enables dropout

    ce_losses = AverageMeter()  # cross entropy loss
    vb_losses = AverageMeter()  # viterbi loss
    f1s = AverageMeter()  # f1 score

    start = time.time()

    # Batches
    for i, (wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths) in enumerate(
            train_loader):

        max_word_len = max(wmap_lengths.tolist())
        max_char_len = max(cmap_lengths.tolist())

        # Reduce batch's padded length to maximum in-batch sequence
        # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
        wmaps = wmaps[:, :max_word_len].to(device)
        cmaps_f = cmaps_f[:, :max_char_len].to(device)
        cmaps_b = cmaps_b[:, :max_char_len].to(device)
        cmarkers_f = cmarkers_f[:, :max_word_len].to(device)
        cmarkers_b = cmarkers_b[:, :max_word_len].to(device)
        tmaps = tmaps[:, :max_word_len].to(device)
        wmap_lengths = wmap_lengths.to(device)
        cmap_lengths = cmap_lengths.to(device)
        
        # Forward prop.
        crf_scores, lm_f_scores, lm_b_scores, wmaps_sorted, tmaps_sorted, wmap_lengths_sorted, _, __ = model(
            cmaps_f,
            cmaps_b,
            cmarkers_f,
            cmarkers_b,
            wmaps,
            tmaps,
            wmap_lengths, 
            cmap_lengths)

        # LM loss

        # We don't predict the next word at the pads or <end> tokens
        # We will only predict at [dunston, checks, in] among [dunston, checks, in, <end>, <pad>, <pad>, ...]
        # So, prediction lengths are word sequence lengths - 1
        lm_lengths = wmap_lengths_sorted - 1
        lm_lengths = lm_lengths.tolist()

        # Remove scores at timesteps we won't predict at
        # pack_padded_sequence is a good trick to do this (see dynamic_rnn.py, where we explore this)
        lm_f_scores = pack_padded_sequence(lm_f_scores, lm_lengths, batch_first=True)
        lm_b_scores = pack_padded_sequence(lm_b_scores, lm_lengths, batch_first=True)

        # For the forward sequence, targets are from the second word onwards, up to <end>
        # (timestep -> target) ...dunston -> checks, ...checks -> in, ...in -> <end>
        lm_f_targets = wmaps_sorted[:, 1:]
        lm_f_targets = pack_padded_sequence(lm_f_targets, lm_lengths, batch_first=True)

        # For the backward sequence, targets are <end> followed by all words except the last word
        # ...notsnud -> <end>, ...skcehc -> dunston, ...ni -> checks
        lm_b_targets = torch.cat(
            [torch.LongTensor([word_map['<end>']] * wmaps_sorted.size(0)).unsqueeze(1).to(device), wmaps_sorted], dim=1)
        lm_b_targets = pack_padded_sequence(lm_b_targets, lm_lengths, batch_first=True)

        # Calculate loss
        ce_loss = lm_criterion(lm_f_scores.data, lm_f_targets.data) + lm_criterion(lm_b_scores.data, lm_b_targets.data)
        wmap_lengths_sorted = wmap_lengths_sorted.to('cpu')
        vb_loss = crf_criterion(crf_scores, tmaps_sorted, wmap_lengths_sorted)
        loss = ce_loss + vb_loss

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()

        # Viterbi decode to find accuracy / f1
        decoded = vb_decoder.decode(crf_scores.to("cpu"), wmap_lengths_sorted.to("cpu"))

        # Remove timesteps we won't predict at, and also <end> tags, because to predict them would be cheating
#         decoded = pack_padded_sequence(decoded, lm_lengths, batch_first=True)
        decoded_tmp = decoded.data.to("cpu").numpy()
        for j, length in enumerate((wmap_lengths_sorted - 1).tolist()):
            decoded_tmp[j][length] = tag_map['<end>']
            decoded_tmp[j][length+1:] = tag_map['<pad>'] * (299 - length)
        decoded = torch.LongTensor(decoded_tmp)
        
        tmaps_sorted = tmaps_sorted % vb_decoder.tagset_size  # actual target indices (see create_input_tensors())      
#         tmaps_sorted = pack_padded_sequence(tmaps_sorted, lm_lengths, batch_first=True)
        tmaps_sorted_tmp = tmaps_sorted.data.to("cpu").numpy()
        for j, length in enumerate((wmap_lengths_sorted - 1).tolist()):
            tmaps_sorted_tmp[j][length] = tag_map['<end>']
            tmaps_sorted_tmp[j][length+1:] = tag_map['<pad>'] * (299 - length)
#         tmaps_sorted = torch.LongTensor(tmaps_sorted_tmp).to(device)
 
        # F1
        y_true = tmaps_sorted.data.to("cpu").numpy()
        y_pred = decoded.data.numpy()

        m = MultiLabelBinarizer().fit(y_true)
        f1 = f1_score(m.transform(y_true), m.transform(y_pred), average='macro')

        # Keep track of metrics
        ce_losses.update(ce_loss.item(), sum(lm_lengths))
        vb_losses.update(vb_loss.item(), crf_scores.size(0))
        f1s.update(f1, sum(lm_lengths))

    # Print training status
    print(f'Epoch: [{epoch}]\t Epoch Time {time.time() - start} \t'
          f'CE Loss {ce_losses.val:.4f} ({ce_losses.avg:.4f})\t'
          f'VB Loss {vb_losses.val:.4f} ({vb_losses.avg:.4f})\t'
          f'F1 {f1s.val:.3f} ({f1s.avg:.3f})')


def validate(val_loader, model, crf_criterion, vb_decoder, word_reflect_map, tag_to_cat, tag_map, ispredicting=False):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data
    :param model: model
    :param crf_criterion: viterbi loss layer
    :param vb_decoder: viterbi decoder
    :return: validation F1 score
    """
    model.eval()

    batch_time = AverageMeter()
    vb_losses = AverageMeter()
    f1s = AverageMeter()
    Preds, Preds_cats, Lbs, Lbs_cats = [], [], [], []

    start = time.time()

    for i, (wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths) in enumerate(
            val_loader):

        max_word_len = max(wmap_lengths.tolist())
        max_char_len = max(cmap_lengths.tolist())

        # Reduce batch's padded length to maximum in-batch sequence
        # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
        wmaps = wmaps[:, :max_word_len].to(device)
        cmaps_f = cmaps_f[:, :max_char_len].to(device)
        cmaps_b = cmaps_b[:, :max_char_len].to(device)
        cmarkers_f = cmarkers_f[:, :max_word_len].to(device)
        cmarkers_b = cmarkers_b[:, :max_word_len].to(device)
        tmaps = tmaps[:, :max_word_len].to(device)
        wmap_lengths = wmap_lengths.to(device)
        cmap_lengths = cmap_lengths.to(device)
        
#         print([word_reflect_map[ws] for ws in wmaps[0].to("cpu").numpy()])
        # Forward prop.
        crf_scores, wmaps_sorted, tmaps_sorted, wmap_lengths_sorted, word_sort_ind, char_sort_ind = model(cmaps_f,
                                                                                   cmaps_b,
                                                                                   cmarkers_f,
                                                                                   cmarkers_b,
                                                                                   wmaps,
                                                                                   tmaps,
                                                                                   wmap_lengths,
                                                                                   cmap_lengths)
#         print(word_sort_ind)
        # Viterbi / CRF layer loss
        vb_loss = crf_criterion(crf_scores, tmaps_sorted, wmap_lengths_sorted.to("cpu"))

        # Viterbi decode to find accuracy / f1
        decoded = vb_decoder.decode(crf_scores.to("cpu"), wmap_lengths_sorted.to("cpu"))

        # Remove timesteps we won't predict at, and also <end> tags, because to predict them would be cheating
#         decoded = pack_padded_sequence(decoded, (wmap_lengths_sorted - 1).tolist(), batch_first=True)
        decoded_tmp = decoded.data.to("cpu").numpy()
        for j, length in enumerate((wmap_lengths_sorted - 1).tolist()):
            decoded_tmp[j][length] = tag_map['<end>']
            decoded_tmp[j][length+1:] = tag_map['<pad>'] * (299 - length)
        decoded = torch.LongTensor(decoded_tmp)
#         print(tmaps_sorted[2])
#         print(tmaps[char_sort_ind][word_sort_ind][2])
        tmaps_sorted = tmaps_sorted % vb_decoder.tagset_size  # actual target indices (see create_input_tensors())
        
        tmaps_sorted_tmp = tmaps_sorted.data.to("cpu").numpy()
#         print(tmaps_sorted_tmp[0])
        for j, length in enumerate((wmap_lengths_sorted - 1).tolist()):
            tmaps_sorted_tmp[j][length] = tag_map['<end>']
            tmaps_sorted_tmp[j][length+1:] = tag_map['<pad>'] * (299 - length)
        tmaps_sorted = torch.LongTensor(tmaps_sorted_tmp).to(device)
#         tmaps_sorted = pack_padded_sequence(tmaps_sorted, (wmap_lengths_sorted - 1).tolist(), batch_first=True)
#         print([word_reflect_map[ws] for ws in wmaps_sorted[2].to("cpu").numpy()])
#         print([word_reflect_map[ws] for ws in wmaps[char_sort_ind][word_sort_ind][2].to("cpu").numpy()])
        
        # f1
        y_true = tmaps_sorted.data.to("cpu").numpy()
        y_pred = decoded.data.numpy()

        m = MultiLabelBinarizer().fit(y_true)
        f1 = f1_score(m.transform(y_true), m.transform(y_pred), average='macro')
        
        if not ispredicting:
            
#             labels.append(tmaps_sorted[word_sort_ind][char_sort_ind].data)
#             predictions.append(decoded[word_sort_ind][char_sort_ind].data)
            flatten_words = []
            for wps in wmaps[char_sort_ind][word_sort_ind].to("cpu").numpy():
                flatten_words.append([word_reflect_map[wp] for wp in wps])
            
            flatten_preds, flatten_labels = [], []
            
            index, total = 0, 0
            preds = decoded.data.numpy().tolist()
            while total < len(preds):
                flatten_preds.extend(preds[total: total+wmap_lengths_sorted[index]])
                total += wmap_lengths_sorted[index]
                index += 1
                
            index, total = 0, 0
            lbs = tmaps_sorted.data.to("cpu").numpy().tolist()
            while total < len(lbs):
                flatten_labels.extend(lbs[total: total+wmap_lengths_sorted[index]])
                total += wmap_lengths_sorted[index]
                index += 1
#             print(flatten_preds[0])
            for i, label_seq in enumerate(flatten_preds):
                pred_labels = [tag_to_cat[t] for t in label_seq]
                label = [tag_to_cat[t] for t in flatten_labels[i]]
                preds, preds_cats = tags_to_keywords(pred_labels, flatten_words[i])
                lbs, lbs_cats = tags_to_keywords(label, flatten_words[i])

                Preds.append(preds)
                Preds_cats.append(preds_cats)
                Lbs.append(lbs)
                Lbs_cats.append(lbs_cats)
                
#             word_orders.append(word_sort_ind.data)
        # Keep track of metrics
        vb_losses.update(vb_loss.item(), crf_scores.size(0))
        f1s.update(f1, sum((wmap_lengths_sorted - 1).tolist()))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'VB Loss {vb_loss.val:.4f} ({vb_loss.avg:.4f})\t'
                  'F1 Score {f1.val:.3f} ({f1.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                  vb_loss=vb_losses, f1=f1s))

    print(
        '\n * LOSS - {vb_loss.avg:.3f}, F1 SCORE - {f1.avg:.3f}\n'.format(vb_loss=vb_losses,
                                                                          f1=f1s))

    return f1s.avg, Preds, Preds_cats, Lbs, Lbs_cats

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

directory_file = '/workspaces/KeywordExtraction/'
data_file = directory_file + 'data/finalData/'
save_file = directory_file + 'models/results/'

emb_file = directory_file + 'originalData/glove.6B.100d.txt'

model_name = 'BiLSTM-CRF'
min_word_freq = 3  # threshold for word frequency
min_char_freq = 1  # threshold for character frequency
caseless = True  # lowercase everything?
expand_vocab = True  # expand model's input vocabulary to the pre-trained embeddings' vocabulary?

# Model parameters
char_emb_dim = 30  # character embedding size
with open(emb_file, 'r') as f:
    word_emb_dim = len(f.readline().split(' ')) - 1  # word embedding size
word_rnn_dim = 300  # word RNN size
char_rnn_dim = 300  # character RNN size
char_rnn_layers = 1  # number of layers in character RNN
word_rnn_layers = 1  # number of layers in word RNN
highway_layers = 1  # number of layers in highway network
dropout = 0.5  # dropout
fine_tune_word_embeddings = False  # fine-tune pre-trained word embeddings?

# Training parameters
start_epoch = 0  # start at this epoch
batch_size = 32  # batch size
lr = 0.015  # learning rate
lr_decay = 0.05  # decay learning rate by this amount
momentum = 0.9  # momentum
workers = 1  # number of workers for loading data in the DataLoader
epochs = 12  # number of epochs to run without early-stopping
grad_clip = 5.  # clip gradients at this value
print_freq = 100  # print training or validation status every __ batches
best_f1 = 0.  # F1 score to start with
checkpoint = None  # path to model checkpoint, None if none

for trainingSession in range(1, 5):

    with open(data_file + f"trainset-{trainingSession}.pkl", "rb") as f:
        train_set = pickle.load(f)
    with open(data_file + f"testset-{trainingSession}.pkl", "rb") as f:
        test_set = pickle.load(f)

    train_set = train_set[:1000]
    test_set = test_set[:1000]
        
    def tokenize_example(example, max_length=300):
        return example.split(' ')[:max_length]

    def split_tags(example, max_length=300):
        return example.split(',')[:max_length]

    # for trainingSession in range(1,5):
    train_set['tokens'] = train_set['sentence'].map(tokenize_example)
    train_set['labels'] = train_set['word_labels'].map(split_tags)
    test_set['tokens'] = test_set['sentence'].map(tokenize_example)
    test_set['labels'] = test_set['word_labels'].map(split_tags)

    data2 = pd.concat([train_set, test_set])
    word_map, char_map, tag_map = create_maps(data2.tokens, data2.labels, min_word_freq,
                                                    min_char_freq)
    embeddings, word_map, lm_vocab_size = load_embeddings(emb_file, word_map,
                                                                expand_vocab)
    model = LM_LSTM_CRF(tagset_size=len(tag_map),
                        charset_size=len(char_map),
                        char_emb_dim=char_emb_dim,
                        char_rnn_dim=char_rnn_dim,
                        char_rnn_layers=char_rnn_layers,
                        vocab_size=len(word_map),
                        lm_vocab_size=lm_vocab_size,
                        word_emb_dim=word_emb_dim,
                        word_rnn_dim=word_rnn_dim,
                        word_rnn_layers=word_rnn_layers,
                        dropout=dropout,
                        highway_layers=highway_layers).to(device)
    model.init_word_embeddings(embeddings.to(device))  # initialize embedding layer with pre-trained embeddings
    model.fine_tune_word_embeddings(fine_tune_word_embeddings)  # fine-tune
    optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum)

    # Loss functions
    lm_criterion = nn.CrossEntropyLoss().to(device)
    crf_criterion = ViterbiLoss(tag_map).to(device)

    # Since the language model's vocab is restricted to in-corpus indices, encode training/val with only these!
    # word_map might have been expanded, and in-corpus words eliminated due to low frequency might still be added because
    # they were in the pre-trained embeddings
    temp_word_map = {k: v for k, v in word_map.items() if v <= word_map['<unk>']}
    train_inputs = create_input_tensors(train_set.tokens, train_set.labels, temp_word_map, char_map,
                                        tag_map)
    val_inputs = create_input_tensors(test_set.tokens, test_set.labels, temp_word_map, char_map, tag_map)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(WCDataset(*train_inputs), batch_size=batch_size, shuffle=True,
                                            num_workers=workers, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(WCDataset(*val_inputs), batch_size=batch_size, shuffle=False,
                                            num_workers=workers, pin_memory=False)

    # Viterbi decoder (to find accuracy during validation)
    vb_decoder = ViterbiDecoder(tag_map)

    word_reflect_map = {v:k for k,v in word_map.items()}
    tag_to_cat = {v:k for k, v in tag_map.items()}

    print(f"{model_name} starts: ")
    for epoch in range(start_epoch, epochs):

        # One epoch's training
        train(train_loader=train_loader,
            model=model,
            lm_criterion=lm_criterion,
            crf_criterion=crf_criterion,
            optimizer=optimizer,
            epoch=epoch,
            vb_decoder=vb_decoder,
            word_map=word_map,
            tag_map=tag_map)

        # One epoch's validation
        val_f1, _, _, _, _ = validate(val_loader=val_loader,
                                    model=model,
                                    crf_criterion=crf_criterion,
                                    vb_decoder=vb_decoder,
                                    ispredicting=True,
                                    word_reflect_map=word_reflect_map,
                                    tag_to_cat=tag_to_cat,
                                    tag_map=tag_map)

        # Did validation F1 score improve?
        is_best = val_f1 > best_f1
        best_f1 = max(val_f1, best_f1)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Decay learning rate every epoch
        adjust_learning_rate(optimizer, lr / (1 + (epoch + 1) * lr_decay))

    tag_to_cat = {v:k for k, v in tag_map.items()}
    _, Preds, Preds_cats, Lbs, Lbs_cats = validate(val_loader=val_loader, 
                                                model=model,
                                                crf_criterion=crf_criterion,
                                                vb_decoder=vb_decoder,
                                                word_reflect_map=word_reflect_map,
                                                tag_to_cat=tag_to_cat,
                                                tag_map=tag_map)

    pred_lable_keywords = {}
    pred_lable_keywords['predicitons'] = Preds
    pred_lable_keywords['preds_cats'] = Preds_cats
    pred_lable_keywords['labels'] = Lbs
    pred_lable_keywords['labels_cats'] = Lbs_cats

    with open(save_file + f"./{model_name}-label-pred-keywords-{trainingSession}.pkl", "wb") as f:
        pickle.dump(pred_lable_keywords, f)

    torch.save(model, save_file + f"./{model_name}-model-{trainingSession}.pt")
