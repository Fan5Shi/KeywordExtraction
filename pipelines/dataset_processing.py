import re
import os
import sys
import ast
import time
import codecs
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from functools import reduce
from tqdm import tqdm

import warnings  
warnings.filterwarnings('ignore')

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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# !pip install openpyxl
import openpyxl

# !pip install python-crfsuite
import pycrfsuite

# !pip install transformers
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from transformers import RobertaForTokenClassification, RobertaTokenizerFast
from transformers import AlbertForTokenClassification, AlbertTokenizerFast

# !pip install sentence-transformers
from sentence_transformers import SentenceTransformer

import torch
from transformers import BertTokenizer, BertModel
import logging
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

seed = 442
np.random.seed(seed)
torch.manual_seed(seed)

# !pip install -U spacy
import spacy
spacy.__version__

from tkinter import Text
from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
import copy
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tag_dict = {}
tag_dict['Sustainability preoccupations'] = 'I-sus'
tag_dict['Digital transformation'] = 'I-dig'
tag_dict['Change in management'] = 'I-mag'
tag_dict['Innovation activities'] = 'I-inn'
tag_dict['Business Model'] = 'I-bus'
tag_dict['Corporate social responsibility ou CSR'] = 'I-cor'
# tag_dict['marco-label'] = 'I-mar'
tag2cat = {v: k for k, v in tag_dict.items()}

labels_to_ids2 = {'O':0, 'I-sus':1, 'I-dig':2, 'I-mag':3, 'I-inn':4, 'I-bus':5, 'I-cor':6}
ids_to_labels2 = {v: k for k, v in labels_to_ids2.items()}

"""
This is the class to read dataset

Return value:
    dataset of the corpus
"""
class datasetReader:
    def __init__(self,
                 dataset_source_name,
                 filename,
                 directory_file="/content/drive/MyDrive/COURSE/Intern/"):
        self.dataset_source_name = dataset_source_name
        self.filename = filename
        self.directory_file = directory_file

        if self.directory_file[-1] != '/':
            self.directory_file += '/'
        
        print(f"Read the dataset from {self.dataset_source_name}")
        if '.csv' in self.filename:
            self.datasetdf = pd.read_csv(self.directory_file+self.filename, sep="\t")
        if '.pkl' in self.filename:
            with open(self.directory_file+self.filename, 'rb') as f:
                self.datasetdf = pickle.load(f)
        self.datasetdf.drop(self.datasetdf.columns[self.datasetdf.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    
    def get_datasetdf(self):
        return self.datasetdf

"""
This is the class to process dataset, first is to split the phrases, then is to label the terms

Return values:

Parameters:

"""
class datasetProcessPipeline:
    def __init__(self, 
                 dataset_source_name,
                 datasetdf,
                 process_filename,
                 col_name,
                 gold_dict,
                 directory_file="/content/drive/MyDrive/COURSE/Intern/",
                 acceptHyphen=True,
                 enriched_dict_file=None):
        self.dataset_source_name = dataset_source_name
        self.process_filename = process_filename
        self.datasetdf = datasetdf
        self.directory_file = directory_file
        self.col_name = col_name
        self.gold_dict = gold_dict
        self.acceptHyphen = acceptHyphen
        self.enrich_dict_file = enriched_dict_file
        self.enrich2cat_dict = None
        self.base_dict = self.gold_dict

        if self.directory_file[-1] != '/':
            self.directory_file += '/'
        
        print(f"Process the dataset from {self.dataset_source_name}")
        if '.csv' in self.process_filename:
            self.datasetdf = pd.read_csv(self.directory_file+self.process_filename)
        if '.pkl' in self.process_filename:
            with open(self.directory_file+self.process_filename, 'rb') as f:
                self.datasetdf = pickle.load(f)
        self.datasetdf.drop(self.datasetdf.columns[self.datasetdf.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        
    def split_sentence(self):
        if not 'label_phrase_spacy' in list(self.datasetdf.columns):
            print("To split the sentence, this might take few hours ... ")
            self.datasetdf['label_phrase_spacy'] = self.datasetdf[self.col_name].map(self.label_phrase_spacy)
    
    def label_terms(self):
        # if not 'label_word_new' in list(self.datasetdf.columns):
        self.datasetdf['label_word_new'] = self.datasetdf['label_phrase_spacy'].map(self.label_word)

    def label_word_new_dict(self, phrases):
        texts = phrases.split('\n')
        categories, keys, enriched_keys = [], [], []
        for text in texts:
            for key, value in self.gold_dict.items():
                matches = re.finditer(r"((?:\w+[-])*(" + key + r")(?:-\w+)*)\b(?: |\.|\,|\:|\?|\!)", text, re.MULTILINE)
                for matchNum, match in enumerate(matches, start=1):
                    start_idx = match.start()
                    end_idx = match.end()
                    enriched_key = match.group(1)
                    key = match.group(2)

                    categories.extend(value)
                    keys.append(key)
                    enriched_keys.append(enriched_key)
        return categories, keys, enriched_keys

    def get_enriched_dict(self):
        if self.enrich_dict_file:
            with open(self.directory_file+self.enrich_dict_file, "rb") as f:
                tmp_dict = pickle.load(f)
                
            Categories = tmp_dict['cats'] 
            Enriched_keys = tmp_dict['kws']
        else:
            print("To get enriched dict, this might take 10 hours ... ")
            Categories, Keys, Enriched_keys = [],[],[]
            for i in tqdm(range(len(self.datasetdf))):
                categories, keys, enriched_keys = self.label_word_new_dict(self.datasetdf['label_phrase_spacy'].iloc[i])
                Enriched_keys.append(enriched_keys)
                # Keys.append(keys)
                Categories.append(categories)
            
            tmp_dict = {'cats': Categories, 'kws': Enriched_keys}
            with open(self.directory_file+f"enriched_cat_{self.dataset_source_name}.pkl", "wb") as f:
                pickle.dump(tmp_dict, f)

        enrich2cat_dict = {}
        for i, key in enumerate(Enriched_keys):
            enrich2cat_dict[key] = Categories[i]
        self.enrich2cat_dict = enrich2cat_dict


    """
    This function is to split the sentence by using spacy package
    The splitted sentences will be tagged as '<phrase>...</phrase>\n'
    """
    def label_phrase_spacy(self, text):
        phrases_xml = ""
        phrases = []
        if len(text) >= 1000000:
            index, gap = 0, 1000000
            while index <= len(text):
                subtext = text[index:min(index+gap, len(text))]
                doc = nlp(subtext)
                phrases.extend([str(sent) for sent in doc.sents])
                index += gap
        else:
            doc = nlp(text)
            phrases = [str(sent) for sent in doc.sents]
        for phrase in phrases[:]:
            if len(phrase) > 0:
                phrases_xml = phrases_xml+"<phrase>"+phrase.strip()+"</phrase>"+"\n"
        return phrases_xml[:-1]

    """
    This function is to label the terms
    """
    # label the terms
    def label_word(self, phrases):
        texts = phrases.split('\n')
        new_texts = []
        for text in texts:
            flag = False
            categories, keys = [], []
            for key, value in self.base_dict.items():	
                if key in text:
                    # detect the case: 
                    # 'product development' in the dictionary
                    # but 'product developments' in the text
                    start_idx = text.index(key)
                    end_idx = text.index(key) + len(key)
                    if bool(re.match('^[a-zA-Z0-9]*$', text[end_idx])): continue
                    text = text[:start_idx]+'<mot '+'category=\''+value+'\'>'+text[start_idx:end_idx]+'</mot>'+text[end_idx:]
                    categories.append(value)
                    keys.append(key)
                    flag = True
            if not flag:
                new_texts.append(text)
                continue
            str_text = copy.deepcopy(text[:7])
            end_text = copy.deepcopy(text[7:])
            new_texts.append(str_text+' category=\''+','.join(set(categories))+'\' values=\''+','.join(set(keys))+'\''+end_text)
        return '\n'.join(new_texts)

    def get_datasetdf(self, plain=True):
        if plain:
            return self.datasetdf
        self.split_sentence()
        
        if self.acceptHyphen:
            self.get_enriched_dict()
            self.base_dict = self.enrich2cat_dict

        self.base_dict = dict(sorted(self.base_dict.items(),key=lambda x:len(x[0]), reverse=True))
        self.label_terms()
        return self.datasetdf

"""
This function is to build the expert terms dictionary from the excel the experts provide

Return value:
The structure of the dictionary is like {'terms': [category the term belongs to]}

    {'academic institutions': ['Innovation activities'],
      'additive manufacturing': ['Digital transformation'], ...}

Parameters:
- canonical: a boolean value, True by default, means to keep the original terms that experts without changing anything;
            False means to enrich the dictionary by adding the inflected form of all the terms, refering to an existing 
            project named Cortext
- directory_file: the directory contains the expert terms file and cortext file
"""
def build_gold_dict(canonical=True,
                    directory_file='/content/drive/MyDrive/COURSE/Intern/',
                    expert_term_file='Terms malantin 1er juin 2022.xlsx',
                    sheet_name='categories 1 juin 2022',
                    cortext_file='Cortext3_min_delac_flex_utf8.txt'):
    # read from excel
    if directory_file[-1] != '/':
        directory_file += '/'
        
    read_file = pd.read_excel(directory_file+expert_term_file, sheet_name=sheet_name)
    read_file.dropna(0, how='all', inplace=True)
    read_file.dropna(1, how='all', inplace=True)

    gold_dict = {}
    for i in range(1, len(read_file)):
        if read_file.iloc[i]['Main form'] is None:
            continue
        gold_dict[read_file.iloc[i]['Main form'].strip()] = []

    for i in range(1, len(read_file)):
        index = read_file.iloc[i]['Main form'].strip()
        if index is None:
            continue
        if not read_file.iloc[i].isna()['Sustainability preoccupations']:
            gold_dict[index].append('Sustainability preoccupations')
        if not read_file.iloc[i].isna()['Digital transformation']:
            gold_dict[index].append('Digital transformation')
        if not read_file.iloc[i].isna()['Change in management']:
            gold_dict[index].append('Change in management')
        if not read_file.iloc[i].isna()['Innovation activities']:
            gold_dict[index].append('Innovation activities')
        if not read_file.iloc[i].isna()['Business Model']:
            gold_dict[index].append('Business Model')
        if not read_file.iloc[i].isna()['Corporate social responsibility ou CSR']:
            gold_dict[index].append('Corporate social responsibility ou CSR') 

    # Change the category for four keywords
    # academic institutions, university & research institutions, service among university, 
    # worldwide research centres
    category = 'Innovation activities'
    changelist = ['academic institutions', 'worldwide research centers', 'university and research institutions', 'customer service among university']
    for c in changelist:
        gold_dict[c][0] = category

    # Deal with the singular and plural cases in keywords
    if not canonical:
        cortext3 = open(directory_file+cortext_file, "r")
        lines = cortext3.readlines()
        lefts, rights = [], []
        for line in lines:
            left, right = line.split(',')
            if left != right.split('.')[0]:
                lefts.append(left)
                rights.append(right.split('.')[0])

        tmp_gold_dict = gold_dict.copy()
        for key, value in tmp_gold_dict.items():
            if key in lefts:
                index = lefts.index(key)
                right = rights[index]
                gold_dict[right] = gold_dict[key]
            if key in rights:
                indices = [i for i, word in enumerate(rights) if word == key]
                for index in indices:
                    left = lefts[index]
                    gold_dict[left] = gold_dict[key]

        tmp_dict = {}
        for k in gold_dict.keys():
            tmp_dict[k] = []
        for k, v in gold_dict.items():
            tmp_dict[k] = list(set(v))
        gold_dict = tmp_dict

    print(f"The size of the expert terms dictionary is {len(gold_dict)}")
    return gold_dict

      
'''
This class is to build our specialized dataset; For the texts that are sentences-splitted and terms-labeled,
blocks are built under the pattern that use the sentence has terms as the center and pick certain numbers of 
sentences ahead and behind it, like a semantically subtext

Return value: the block-based dataset with the previous information

Parameters:
- filepath: None by default, the file path of the previous processed dataset, must be assigned a valid value if datasetdf is None
- datasetdf: None by default, the previous processed dataset, must be assigned a valid value if filepath is None
- lookahead/lookbehind: 2 by default, should be assigned any number greater than 0
- limit: True represents taking the value of lookahead and lookbehind 
              as the upper limit to construction the blocks
          False represents taking the value of lookahead and lookbehind 
              as the lower limit to construction the blocks
- output: None by default, means not to store the dataset as an output
- verbose: False by default, if set it to True, will keep all the columns from the previous processed dataset
- fixedLength: False represents generating the blocks based on the sentences
                True represents generating the blocks based on the number of tokens, 
                works only set up the number of tokens per block at the same time
- numTokens: 200 by default
- colnames: None by default, if verbose is True, colnames will be the list of all the columns of the processed dataset
            otherwise, it could be customized and it's better to customized due to the different data structure
- enrich2cat_dict: enriched terms
'''
class DatasetBuilder:
    def __init__(self, dataset_source_name, filepath=None, datasetdf=None, lookahead=2, lookbehind=2, limit=True, output=None, verbose=False, fixedLength=False, numTokens=200, colnames=None, enrich2cat_dict=Nonde):
        self.dataset_source_name = dataset_source_name
        self.filepath = filepath
        self.datasetdf = datasetdf
        self.lookahead = lookahead
        self.lookbehind = lookbehind
        self.verbose = verbose
        self.limit = limit
        self.output = output
        self.fixedLength = fixedLength
        self.numTokens = numTokens
        self.colnames = colnames
      
        self.allsents_cat = None
        self.sourcedf = None
        self.sents_labels_dict = None
        self.sents_labels_blocks = None
        self.dataset = None

        self.enrich2cat_dict = enrich2cat_dict
    
    def get_sents(self, col_name='label_word_new'):
        if not self.datasetdf is None:
            df = self.datasetdf
        else:
            # read file
            if self.filepath.endswith(".csv"):
                df = pd.read_csv(self.filepath)
            else: # endswith pkl
                with open(self.filepath, "rb") as f:
                    df = pickle.load(f)
        # store sentences
        allsents_cat = {}
        for i in range(len(df)):
            allsents_cat[i] = df[col_name].iloc[i].split('\n')
        self.sourcedf = df
        self.allsents_cat = allsents_cat
    
    def build_fixed_tokens(self, numTokens=200):
        if self.numTokens != numTokens:
            self.numTokens = numTokens
        
        # build the dictionary
        sents_labels_blocks = {}
        for key, sents in self.allsents_cat.items():
            string = ' '.join(sents)
            string = re.sub(r'\scategory=\'[^\']+\'\svalues=\'[^\']+\'', '', string)
            string = re.sub(r'\scategory=\'[^\']+\'', '', string)
            # print(string)
            stringlist = re.findall("(\S+\s\S+\s<mot>[^\']+?<\/mot>)", string)
            # print(stringlist)
            results = []
            for l in stringlist:
                index = string.index(l)
                left = string[:index]
                right = string[index+len(l):]
                
                if len(left.split()) >= (self.numTokens - 2):
                    left = ' '.join(left.split()[-(self.numTokens - 2):])
                if len(right.split()) >= self.numTokens:
                    right = ' '.join(right.split()[-self.numTokens:])

                result = left + l + right
                result = re.sub("<phrase>", "", result)
                result = re.sub("</phrase>", "", result)
                keywords = re.findall("<mot>(.+?)</mot>", result)
                keywords = set(keywords)
                # print(keywords)
                categories = []
                real_keywords = []
                for kw in keywords:
                    # problems need to be solved
                    try:
                        categories.append(self.enrich2cat_dict[kw][0])
                        real_keywords.append(kw)
                    except:
                        continue
                results.append((result, set(categories), real_keywords))    
            sents_labels_blocks[key] = results
        self.sents_labels_blocks = sents_labels_blocks                    

    def build_fixed_windows(self):
        # first, build a dictionary like:
        # key is the row index in the corpus
        # value consists of the index of the sentence, the label the sentence has, keywords the sentence has
        # dict = {
        #     "1": [(1, label, keywords), (2, label, keywords), ...],
        #     "2": [(1, label, keywords), (2, label, keywords), ...]...
        # }
        sents_labels_dict = {}
        for key, sents in self.allsents_cat.items():
            sents_labels = []
            for i, sent in enumerate(sents):
                labels = []
                keywords = []
                if 'category=' in sent:
                    labels = re.findall(r'<phrase category=\'([^\']+)', sent)
                    # if match: labels = match.group(1)
                    keywords = re.findall(r'values=\'([^\']+)\'>', sent)
                sents_labels.append((i, labels, keywords))
            sents_labels_dict[key] = sents_labels
        self.sents_labels_dict = sents_labels_dict

        # second, build a dictionary like:
        # key is the row index in the corpus
        # value consists of the indices of the sentences block, a list
        #                   the labels the block has, a set
        #                   the keywords the block contains, a list
        # only the sentence has at least a label can be considered to build the block surround it
        # dict = {
        #     "1": [(indices, labels, keywords), (indices, labels, keywords), ...],
        #     "2": [(indices, labels, keywords), (indices, labels, keywords), ...]...
        # }
        sents_labels_blocks = {}
        if self.limit:
            for key, sentlabs in sents_labels_dict.items():
                indices_blocks = []
                for i, label, keywords in sentlabs:
                    if label:
                        start_idx = 0 if i - self.lookahead <= 0 else i - self.lookahead
                        end_idx = len(sentlabs) if i + self.lookbehind + 1 >= len(sentlabs) else i + self.lookbehind + 1
                        labels, keywords = [], []
                        for _, l, kw in sentlabs[start_idx:end_idx]:
                            if l:
                                labels.extend(l)
                                keywords.extend(kw[0].split(','))
                        indices_blocks.append(((start_idx, end_idx), set(','.join(labels).split(',')), set(keywords)))
                sents_labels_blocks[key] = indices_blocks
            self.sents_labels_blocks = sents_labels_blocks
        else:
            for key, sentlabs in sents_labels_dict.items():
                indices_blocks = []
                global_idx = 0
                for i, label, keywords in sentlabs:
                    if global_idx > i: continue
                    if label:
                        start_idx = 0 if i - self.lookahead <= 0 else i - self.lookahead
                        end_idx = len(sentlabs) if i + self.lookbehind + 1 >= len(sentlabs) else i + self.lookbehind + 1
                        if end_idx is not len(sentlabs):
                            for j in range(end_idx, len(sentlabs)):
                                t, t_label, _ = sentlabs[j]
                                m, m_label, _ = sentlabs[j-1]
                                if not t_label and not m_label:
                                    end_idx = j
                                    global_idx = j
                                    break
                        labels, keywords = [], []
                        for _, l, kw in sentlabs[start_idx:end_idx]:
                            if l:
                                labels.extend(l)
                                keywords.extend(kw[0].split(','))
                        indices_blocks.append(((start_idx, end_idx), set(','.join(labels).split(',')), set(keywords)))
                sents_labels_blocks[key] = indices_blocks
            self.sents_labels_blocks = sents_labels_blocks

    def build(self, col_name):
        # get all the sentences per company
        self.get_sents(col_name)
        
        if self.fixedLength:
            self.build_fixed_tokens(self.numTokens)
        else:
            self.build_fixed_windows()
  
        # third, output the csv file
        new_dataset = pd.DataFrame({})
        rawphrases, phrases, labels, keywords = [], [], [], []
        if self.verbose:
            if self.colnames is None:
                self.colnames = self.sourcedf.columns
            for col in self.colnames:
                tmp_list = []
                for key, values in self.sents_labels_blocks.items():
                    tmp_list.extend([self.sourcedf[col].iloc[key]] * len(values))
                new_dataset[col] = tmp_list

        for key, values in self.sents_labels_blocks.items():
            for sents, labs, keywds in values:
                if self.fixedLength:
                    phrase = sents
                else:
                    start_idx, end_idx = sents
                    phrase = '\n'.join(self.allsents_cat[key][start_idx: end_idx])
                phrases.append(phrase)
                labels.append(','.join(labs))
                rawphrases.append(re.compile(r'<.*?>').sub('', phrase))
                # remove the duplicated keywords - 06.16
                keywds = list(set(keywds))
                keywords.append(','.join(keywds))
        
        new_dataset['Text_para'] = rawphrases
        new_dataset['Text_block'] = phrases
        new_dataset['Catogory'] = labels
        new_dataset['Keyword'] = keywords           

        new_dataset = new_dataset.drop_duplicates(subset=['Text_para','Text_block', 'Catogory', 'Keyword'], keep='first', ignore_index=True)
        if self.output: new_dataset.to_csv(self.output)
        self.dataset = new_dataset
      
    def get_dataset(self):
        print(f"Build the block-based dataset from {self.dataset_source_name}")
        return self.dataset


"""
This class is to trim the block-based dataset, if the block reaches the maximum length;
How to trim the block: cutting off the front and end sub-sentences which are divided by punctuation;
How to measure the length of block: depends on the tokenizer you use, here we use bert tokenzier by default

Return value: the trimmed block-based dataset
"""
class DatasetTrimmer:
    def __init__(self, dataset_source_name, filepath=None, datasetdf=None, max_length=300, verbose=False, colnames=None):
        self.dataset_source_name = dataset_source_name
        self.filepath = filepath
        self.datasetdf = datasetdf
        self.max_length = max_length
        self.verbose = verbose
        self.colnames = colnames

        self.dataset = None

    def get_datasetdf(self):
        if not self.datasetdf is None:
            df = self.datasetdf
        else:
            # read file
            if self.filepath.endswith(".csv"):
                df = pd.read_csv(self.filepath)
            else: # endswith pkl
                with open(self.filepath, "rb") as f:
                    df = pickle.load(f)
        self.dataset = df

    def process(self):
        pass

"""
The inherited class of DatasetTrimmer and BT stands for BertTokenizer
"""
class DatasetTrimmerBT(DatasetTrimmer):
    def __init__(self, dataset_source_name, filepath=None, datasetdf=None, max_length=300, verbose=False, colnames=None):
        super().__init__(dataset_source_name, filepath, datasetdf, max_length, verbose, colnames)

    def get_datasetdf(self):
        super().get_datasetdf()

        def text_tokenizer(text):
            marked_text = "[CLS] " + text + " [SEP]"
            return tokenizer.tokenize(marked_text)

        def keyword_tokenizer(keywords):
            new_keywords = []
            for kw in keywords.split(','):
                new_keywords.append(tokenizer.convert_tokens_to_string(tokenizer.tokenize(kw)))
            return new_keywords

        self.dataset['tokens'] = self.dataset['Text_para'].map(text_tokenizer)
        self.dataset['tokens_len'] = self.dataset['tokens'].map(len)
        self.dataset['tokenized Keywords'] = self.dataset['Keyword'].map(keyword_tokenizer)

    def process(self):
        self.get_datasetdf()

        new_dataset = pd.DataFrame({})
        if self.verbose:
            if self.colnames is None:
                self.colnames = self.dataset.columns
            for col in self.colnames:
                new_dataset[col] = self.dataset[col]

        trimmed_list, keyword_list, tokenized_keyword_list = [],[],[]
        for i in range(len(self.dataset)):
            if self.dataset['tokens_len'].iloc[i] <= self.max_length:
                trimmed_list.append(tokenizer.convert_tokens_to_string(self.dataset['tokens'].iloc[i]))
                keyword_list.append(self.dataset['Keyword'].iloc[i])
                tokenized_keyword_list.append(self.dataset['tokenized Keywords'].iloc[i])
            else:
                ptokens = self.dataset['tokens'].iloc[i]
                pt_lists = [i for i, w in enumerate(ptokens) if w in ',.!?']
                step = 1
                current_len = self.dataset['tokens_len'].iloc[i]
                while current_len >= self.max_length:
                    new_block = ptokens[pt_lists[step-1]+1:pt_lists[-step]+1]
                    current_len = len(new_block)
                    step += 1
                new_block = tokenizer.convert_tokens_to_string(new_block)
                trimmed_list.append(new_block)
            
                keywords = self.dataset['Keyword'].iloc[i].split(',')
                tokenized_keywords = self.dataset['tokenized Keywords'].iloc[i]
                trimmed_keywords, trimmed_tokenized_keywords = [],[]
                for j, kw in enumerate(tokenized_keywords):
                    if kw in new_block:
                        # print(keywords[j])
                        trimmed_keywords.append(keywords[j])
                        trimmed_tokenized_keywords.append(kw)
                keyword_list.append(','.join(trimmed_keywords))
                tokenized_keyword_list.append(trimmed_tokenized_keywords)
        
        new_dataset['tokens'] = trimmed_list
        new_dataset['keywords'] = keyword_list
        new_dataset['tokenized Keywords'] = tokenized_keyword_list

        self.dataset = new_dataset[new_dataset['keywords'].map(len) != 0]

        print(f"Trim the block-based dataset from {self.dataset_source_name}")
        return self.dataset
      

"""
This is the class to filter the labeled keywords, 
the key step to distinguish the learning task from keywords annotation
"""
class KeywordsFilter:
    def __init__(self, filepath, datasetdf):
        self.filepath = filepath
        self.datasetdf = datasetdf

        self.dataset = None
        self.SCORES = None
    
    def get_datasetdf(self):
        if not self.datasetdf is None:
            df = self.datasetdf
        else:
            # read file
            if self.filepath.endswith(".csv"):
                df = pd.read_csv(self.filepath)
            else: # endswith pkl
                with open(self.filepath, "rb") as f:
                    df = pickle.load(f)
        self.dataset = df
    
    def process(self):
        self.get_datasetdf()


"""
This class is inherited from KeywordsFilter, which is a specific filter that uses cosine similarity between 
key terms and the context surrounded(blocks)

Return values: the filtered keywords list of each block, after the filter, only the key words that are semantically close
                to the blocks will be remained

Parameters:
- coef_threshold: 0 by default, it is coefficient in the following formula: threshold = mean - coef_threshold * standard deviation
                  where the mean and std are computed by the distribution of all the labeled keyword
                  It normally range from -3 to +3 and int
"""
class CosineSimFilter(KeywordsFilter):
    def __init__(self, filepath=None, datasetdf=None, coef_threshold=0):
        super().__init__(filepath, datasetdf)
        self.coef_threshold = coef_threshold

        self.mean = None
        self.std = None
        self.threshold = None
        self.filtered_SCORES = None

        self.index_dict = {}

    def store_index_between_text_and_tokenizedText(self, text, t_text, m):
        index_dict = [0] * len(t_text)
        pointer_i, pointer_j = 0, 0
        for i in range(pointer_i, len(text)):
            for j in range(pointer_j, len(t_text)):
                wd = text[i]
                twd = t_text[j]
                if twd == wd:
                    index_dict[j] = i
                    pointer_j += 1
                    pointer_i += 1
                else:
                  if i == len(text)-1:
                      index_dict[j:] = i
                  else:
                      tmp_j = j+1
                      while t_text[tmp_j].startswith('##'):
                          tmp_j += 1
                      index_dict[j:tmp_j] = [i] * (tmp_j-j)
                      pointer_j += tmp_j-j
                      pointer_i += 1
                break
        self.index_dict[m] = index_dict

    def process(self,):
        super().get_datasetdf()

        SCORES = []
        for i in tqdm(range(len(self.dataset))):

            keywords = self.dataset['keywords'].iloc[i].split(',')
            text = self.dataset['tokens'].iloc[i]

            # Tokenize our sentence with the BERT tokenizer.
            tokenized_text = tokenizer.tokenize(text)
            tokenzied_keywords = []
            for keyword in keywords:
                tokenzied_keywords.append(tokenizer.tokenize(keyword))
            
            self.store_index_between_text_and_tokenizedText(text.split(), tokenized_text, i)

            # Map the token strings to their vocabulary indeces.
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

            segments_ids = [1] * len(tokenized_text)

            tokens_tensor = torch.tensor([indexed_tokens]).to(device)
            segments_tensors = torch.tensor([segments_ids]).to(device)

            with torch.no_grad():
                outputs = model(tokens_tensor, segments_tensors)
                hidden_states = outputs[2]

            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings.permute(1,0,2)

            token_vecs_cat = []
            for token in token_embeddings:
                cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
                token_vecs_cat.append(cat_vec)

            token_vecs_sum = []
            for token in token_embeddings:
                sum_vec = torch.sum(token[-4:], dim=0)
                token_vecs_sum.append(sum_vec)

            token_vecs = hidden_states[-2][0]
            sentence_embedding = torch.mean(token_vecs, dim=0)

            Scores = []
            for i, tks in enumerate(tokenzied_keywords):
                start = 0
                end = len(tokenized_text)
                tk_len = len(tks)
                while True:
                    try:
                        index = tokenized_text.index(tks[0], start, end)
                        if tk_len == 1:
                            score = cosine(sentence_embedding.to("cpu"), token_vecs_sum[index].to("cpu"))
        #                     index = len(tokenizer.convert_tokens_to_string(tokenized_text[:index]).split())-1
        #                     tk_len = len(tokenizer.convert_tokens_to_string(tks).split())
                            Scores.append((index, tk_len, score, tks, keywords[i]))
                            start = index + 1

                        if tk_len >= 2:
                            flag = True
                            for span in range(1, tk_len):
                                if tokenized_text[index+span] != tks[span]:
                                    flag = False
                                    break
                            if flag:
                                score = cosine(sentence_embedding.to("cpu"), torch.mean(torch.stack(token_vecs_sum[index:index+len(tks)]), dim=0).to("cpu"))
        #                         index = len(tokenizer.convert_tokens_to_string(tokenized_text[:index]).split())-1
        #                         tk_len = len(tokenizer.convert_tokens_to_string(tks).split())
                                Scores.append((index, tk_len, score, tks, keywords[i]))
                                start = index + 1
                            else:
                                start = index + 1
                                continue
                    except:
                        break
            SCORES.append(Scores)
        
        self.SCORES = SCORES
    
    def get_threshold(self):
        only_scores = []
        for Scores in self.SCORES:
            for _,_,score,_,_ in Scores:
                only_scores.append(score)
        mean = np.mean(np.array(only_scores))
        std = np.std(np.array(only_scores))

        self.mean = mean
        self.std = std

        self.threshold = self.mean + self.coef_threshold * self.std

    def get_filtered_scores(self, coef_threshold=0):
        if self.coef_threshold != coef_threshold:
            self.threshold = self.mean + self.coef_threshold * self.std
        filtered_SCORES = []
        for Scores in self.SCORES:
            tmp_Scores = []
            for score in Scores:
                if score[2] >= self.threshold:
                    tmp_Scores.append(score)
            filtered_SCORES.append(tmp_Scores)
        
        self.filtered_SCORES = filtered_SCORES

        return self.filtered_SCORES

    def get_dataset(self,):
        self.process()
        self.get_threshold()


def DatasetTagger(datasetdf, text_col, myfilter):
    TAGS, TEXTS, KWS = [], [], []
    filtered_SCORES = myfilter.get_filtered_scores(0)
    index_dict = myfilter.index_dict
    for m in range(len(datasetdf)):
        # print(m)
#         texts = ast.literal_eval(datasetdf[text_col].iloc[m])
        texts = datasetdf[text_col].iloc[m].split()
        tags = ['O'] * len(texts)

        keywords = filtered_SCORES[m]
        index_maps = index_dict.get(m)
        Kws = []        
        for n, (index, length_kw, _, _, kw) in enumerate(keywords):
            # if m == 5: print(len(texts), n, index, length_kw, kw, len(kw.split()))
            Kws.append(kw)
            tt = tag_dict[enrich2cat_dict[kw]]
            flag = False
            latter_tt = ''
            length_kw = len(kw.split())
            index = index_maps[index]

            for span in range(length_kw):
                if tags[index+span] != 'O' and tags[index+span] != tt:
                    flag = True
                    latter_tt = tags[index+span]
                    break
            # find boundary of the macro label
            # label the same category as the latter term
            if flag:
                for span in range(index+length_kw, len(texts)):
                    if tags[span] == 'O':
                        length_kw = span-index
                        break
            tags[index:index+length_kw] = [tt] * length_kw
        
        TAGS.append([str(tg) for tg in tags])
        TEXTS.append(texts)
        KWS.append(set(Kws))
    
    datasetdf['word_labels'] = [','.join(bt) for bt in TAGS]
    datasetdf['sentence'] = [' '.join(tx) for tx in TEXTS]
    datasetdf['keywords'] = KWS

    return datasetdf[["sentence", "word_labels", "keywords"]]#.drop_duplicates(subset=["sentence", "word_labels"]).reset_index(drop=True)    


def main():

    # the parameters need to change
    directory_file = '/content/drive/MyDrive/COURSE/Intern/'
    dataset_source_name = 'web' # 'ap'

    gold_dict = build_gold_dict(canonical=False, directory_file='/content/drive/MyDrive/COURSE/Intern/')
    
    filename = 'corpus_v3_per_company_25_04_2021.pkl'
    # filename = "AR_one_per_company_total.csv"
    saved_file = directory_file + 'data/intermediateData/'

    if dataset_source_name == 'web':
        process_filename = "data/intermediateData/processed_corpus_09_27_old.csv"
        col_name = 'Text_para'
        enriched_dict_file = 'ddata/intermediateData/0113_enriched_cat_web.pkl'
        colnames = ['Firmreg_id','Nb_company','Company','Sector']
    else:
        process_filename = "data/intermediateData/processed_corpus_ap_09_27_old.csv"
        col_name = 'Text_basic_clean'
        enriched_dict_file = 'data/intermediateData/0113_enriched_cat_ap.pkl'
        colnames = ['Company','Year','File']

    datasetdf = datasetReader(dataset_source_name=dataset_source_name, 
                              filename=filename, 
                              directory_file=directory_file).get_datasetdf()

    processpl = datasetProcessPipeline(dataset_source_name=dataset_source_name, 
                                       datasetdf=None,
                                       process_filename=process_filename,
                                       col_name=col_name,
                                       gold_dict=gold_dict,
                                       directory_file=directory_file,
                                       enriched_dict_file=enriched_dict_file)
    processdf = processpl.get_datasetdf(False)
    enrich2cat_dict = processpl.enrich2cat_dict
    
    mybuilder = DatasetBuilder(dataset_source_name=dataset_source_name,
                            datasetdf=processdf,
                            lookahead=2,
                            lookbehind=2,
                            limit=True,
                            #  fixedLength=True, 
                            #  numTokens=100,
                            #  output="dataset_07_07_ftoken.csv",
                            verbose=True,
                            colnames=colnames,
                            enrich2cat_dict=enrich2cat_dict)
    mybuilder.build('label_word_new')
    dataset = mybuilder.get_dataset()

    dataset.to_csv(saved_file + f"found_dataset_{dataset_source_name}_01_13.csv")
    dataset = pd.read_csv(saved_file + f"found_dataset_{dataset_source_name}_01_13.csv")
    dataset.drop(dataset.columns[dataset.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    
    colnames.append('Text_block')
    trimmed_dataset = DatasetTrimmerBT(dataset_source_name, 
                                   filepath=None, 
                                   datasetdf=dataset,
                                   verbose=True, 
                                   colnames=colnames).process()
    trimmed_dataset.to_csv(saved_file + f"trimmed_dataset_{dataset_source_name}_01_13.csv")

    model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.)
    model.eval()
    model.to(device)

    myfilter = CosineSimFilter(filepath=None, datasetdf=trimmed_dataset)
    myfilter.get_dataset()

    tagged_dataset = DatasetTagger(trimmed_dataset, "tokens", myfilter)
    tagged_dataset.to_csv(saved_file + f"tagged_dataset_{dataset_source_name}_01_13.csv")  

if __name__ == "__main__":
    main()
