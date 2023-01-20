#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:47:25 2023

@author: revekka
"""
import pandas as pd
from nltk.tokenize import sent_tokenize

index = '4'
unpickled_dico = pd.read_pickle("/Users/revekkakyriakoglou/Documents/paris_8/Projects/KeywordExtraction/models/results/Roberta-label-pred-keywords-"+index+".pkl")  
unpickled_text_blocks = pd.read_pickle("/Users/revekkakyriakoglou/Documents/paris_8/Projects/KeywordExtraction/data/finalData/testset-"+index+".pkl")  

#because the indexing does n ot start from 0, I have to reset it in order to be able to merge the two dataframes later in this code
unpickled_text_blocks.index = pd.RangeIndex(len(unpickled_text_blocks.index))

print(unpickled_dico.keys())
#print(unpickled_text_blocks.keys())

df_one = pd.DataFrame(unpickled_dico)
df_two = pd.DataFrame(unpickled_text_blocks)
print(df_one)
print(df_two)


df_one["sentence"] = df_two["sentence"]
df_one["sentence"] = df_one["sentence"].str[6:-6]
df_one["source"] = df_two["source"]

print(df_one)
df_one.to_pickle("./text_and_predictions-1.pkl") 
df_one.to_csv("./text_and_predictions-1.csv") 

text_list = df_one["sentence"].tolist()
print(len(text_list))
tokenized_text_list = []

for element in text_list:
    tokenized_text_list.append(sent_tokenize(element))

lines_to_be_merged = []
text_concatenated_merged = []
source_merged = []
predicitons_merged = []
preds_cats_merged = []
labels_merged = []
labels_cats_merged = []


for i in range(len(tokenized_text_list)-1):
    if tokenized_text_list[i][-2:] == tokenized_text_list[i+1][0:2]:
        #print("continues: "+str(i))
        lines_to_be_merged.append([i,i+1])
        text_concatenated_merged.append(tokenized_text_list[i]+tokenized_text_list[i+1][2:])
        predicitons_merged.append(df_one.iloc[i]['predicitons'] + list(set(df_one.iloc[i+1]['predicitons']) - set(df_one.iloc[i]['predicitons'])))
        preds_cats_merged.append(df_one.iloc[i]['preds_cats'] + df_one.iloc[i+1]['preds_cats'])
        labels_merged.append(df_one.iloc[i]['labels'] + list(set(df_one.iloc[i+1]['labels']) - set(df_one.iloc[i]['labels'])))
        labels_cats_merged.append(df_one.iloc[i]['labels_cats'] + df_one.iloc[i+1]['labels_cats'])
        source_merged.append(df_one.iloc[i]['source'])
    else:
        lines_to_be_merged.append([i])
        text_concatenated_merged.append(tokenized_text_list[i])
        predicitons_merged.append(df_one.iloc[i]['predicitons'])
        preds_cats_merged.append(df_one.iloc[i]['preds_cats'])
        labels_merged.append(df_one.iloc[i]['labels'])
        labels_cats_merged.append(df_one.iloc[i]['labels_cats'])
        source_merged.append(df_one.iloc[i]['source'])


print(len(lines_to_be_merged)) #list of lists of two integers, that have to be consecutive
print(len(source_merged))
print(len(text_concatenated_merged))
print(len(predicitons_merged))
print(len(preds_cats_merged))
print(len(labels_merged))
print(len(labels_cats_merged))
#print(source_merged)
#print(text_concatenated_merged[0])
#print(text_concatenated_merged[0][0])

"""
#test in order to see tha everything is ok with lines_to_be_merged
compt = 0
for element in lines_to_be_merged:
    if (element[1] == element[0] +1) :
        compt += 1
print(compt)
print('length of list of tuples') 
print(len(lines_to_be_merged))  
"""

           
def concatenate_lists(L):
    i = 0
    while i < len(L) - 1:
        if (len(L[i])!=1) and (L[i][-1] == L[i+1][0]):
            #L[i] = [L[i][0], L[i+1][-1]]
            L[i] = L[i] + list(set(L[i+1]) - set(L[i]))
            del L[i+1]
            text_concatenated_merged[i] = text_concatenated_merged[i] + list(set(text_concatenated_merged[i+1]) - set(text_concatenated_merged[i]))
            del text_concatenated_merged[i+1]
            predicitons_merged[i] = predicitons_merged[i] + list(set(predicitons_merged[i+1]) - set(predicitons_merged[i]))
            del predicitons_merged[i+1]
            preds_cats_merged[i] = preds_cats_merged[i] + preds_cats_merged[i+1]
            del preds_cats_merged[i+1]
            labels_merged[i] = labels_merged[i] + list(set(labels_merged[i+1]) - set(labels_merged[i]))
            del labels_merged[i+1]
            labels_cats_merged[i] = labels_cats_merged[i] + labels_cats_merged[i+1]
            del labels_cats_merged[i+1] 
            del source_merged[i+1] 
            i -= 1
        i += 1
    return L, text_concatenated_merged, predicitons_merged, preds_cats_merged, labels_merged, labels_cats_merged, source_merged

list_final, text_concatenated_merged_final, predicitons_merged_final, preds_cats_merged_final, labels_merged_final, labels_cats_merged_final, source_merged_final = concatenate_lists(lines_to_be_merged)
print("---------------------------------")
#print(list_final) 

#create a dataframe using the new lists
df_merged = pd.DataFrame(list(zip(list_final, text_concatenated_merged_final, predicitons_merged_final, preds_cats_merged_final, labels_merged_final, labels_cats_merged_final, source_merged_final )),
               columns =['blocks', 'text', 'predictions', 'preds_cats', 'labels', 'labels_cats', 'source'])
print(df_merged)

df_merged.to_csv("./merged_blocks_results-"+index+".csv")  



