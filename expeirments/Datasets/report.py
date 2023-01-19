import ast
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def get_recall(Labels, Predictions, ignore=True):
    recalls = []
    for i, lbs in enumerate(Labels):
        preds = Predictions[i]
        if not lbs:
            if not ignore:
                recalls.append(1.0 if not preds else 0.0)
            continue
        recalls.append(len(intersection(preds, lbs)) / len(lbs))
    return np.mean(recalls)

def get_precision(Labels, Predictions, ignore=True):
    precisions = []
    for i, preds in enumerate(Predictions):
        lbs = Labels[i]
        if not preds:
            if not ignore:
                precisions.append(1.0 if not preds else 0.0)
            continue
        precisions.append(len(intersection(preds, lbs)) / len(preds))
    return np.mean(precisions)

def get_f1score(Labels, Predictions, ignore=True):
    precision = get_precision(Labels, Predictions, ignore)
    recall = get_recall(Labels, Predictions, ignore)
    return 2 * precision * recall / (precision + recall)

def get_canonical_labels(labeled_set, cortext_file='Cortext3_min_delac_flex_utf8.txt'):
    cortext3 = open(cortext_dir+cortext_file, "r")
    lines = cortext3.readlines()
    lefts, rights = [], []
    for line in lines:
        left, right = line.split(',')
        lefts.append(left)
        rights.append(right.split('.')[0])

    canoset = []
    for value in labeled_set:
        if value in rights:
            index = rights.index(value)
            left = lefts[index]
            canoset.append(left) 
        else:
            canoset.append(value)

    return set(canoset)

# To see how many new blocks generate new words
def filterAndCosineSim(new_keywords, testset_emb, test_set):
    num = 0 
    new_keywords_filtered = []
    for i, keyword in new_keywords:
        filtered = [kw for kw in keyword if kw.lower() not in gold_dict.keys() and len(kw.split()) <= 5]
        if len(filtered) != 0 :
            new_keywords_filtered.append((i, filtered))

    new_keywords1 = [[]] * len(test_set)
    
    for id, nks in new_keywords_filtered:
        doc_embedding = testset_emb[id]
        candidate_embeddings = model.encode(nks)
        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        new_keywords1[id] = [[nk, distances[0][i]] for i, nk in enumerate(nks)]

    new_keywords0 = [[]] * len(test_set)
    for id, nks in enumerate(test_set['keywords_0.1'].values.tolist()):
        doc_embedding = testset_emb[id]
        candidate_embeddings = model.encode(nks)
        distances = cosine_similarity(doc_embedding, candidate_embeddings)
        new_keywords0[id] = [[nk, distances[0][i]] for i, nk in enumerate(nks)]

    test_set['new_keywords1'] = new_keywords1
    test_set['new_keywords'] = new_keywords0
    sents, originals, news = [], [], []
    new_df = test_set[test_set['new_keywords1'].map(len) != 0][['sentence', 'new_keywords', 'new_keywords1']].reset_index(drop=True)
    for i in range(len(new_df)):
        score_list = new_df['new_keywords'].iloc[i]
        # find the maximum new keyword
        max_cos_new = np.max([kw[1] for kw in new_df['new_keywords1'].iloc[i]])
        max_cos_old = np.max([kw[1] for kw in score_list])
        if max_cos_new > max_cos_old:
            num += 1
            sents_list = new_df['sentence'].iloc[i]
            sents.append(sents_list)
            originals.append(str(score_list))
            news.append(new_df['new_keywords1'].iloc[i])
    return num, sents, originals, news

def get_report(file_path, trainingSession, withTag=True):
    with open(f"{data_path}/testset-{trainingSession}.pkl", "rb") as f:
        test_set = pickle.load(f)
    recalls, precisions, f1scores = [], [], []

    for model_name in models_name:
        if withTag:
            with open(f"{file_path}/{model_name}-label-pred-keywords-{trainingSession}.pkl", "rb") as f:
                pred_lable_keywords = pickle.load(f)
        else:
            with open(f"{file_path}/{model_name}-label-pred-keywords-new-{trainingSession}.pkl", "rb") as f:
                pred_lable_keywords = pickle.load(f)

        predictions = pred_lable_keywords['predicitons']
        labels = pred_lable_keywords['labels']
        # labels = test_set['keywords_0.1'].values.tolist()

        recall = get_recall(labels, predictions, False)
        precision = get_precision(labels, predictions, False)
        f1score = get_f1score(labels, predictions, False)
        recalls.append('{:.3f}'.format(recall))
        precisions.append('{:.3f}'.format(precision))
        f1scores.append('{:.3f}'.format(f1score))

    return recalls, precisions, f1scores

def get_new_terms_amount(file_path, trainingSession, withTag=True):
    with open(f"{data_path}/trainset-{trainingSession}.pkl", "rb") as f:
        train_set = pickle.load(f)
    with open(f"{data_path}/testset-{trainingSession}.pkl", "rb") as f:
        test_set = pickle.load(f)
    train_terms = set([kw for kws in train_set['keywords'].values.tolist() for kw in kws])
    # test_terms = set([kw for kws in test_set['keywords_0.1'].values.tolist() for kw in kws])
    Num_train_terms, Num_test_terms, Num_same_terms, Num_diff_terms, Num_Pred_terms, Num_pred_New_terms, Num_new_terms, New_terms = [],[],[],[],[],[],[],[]
    for model_name in models_name:
        if withTag:
            with open(f"{file_path}/{model_name}-label-pred-keywords-{trainingSession}.pkl", "rb") as f:
                pred_lable_keywords = pickle.load(f)
        else:
            with open(f"{file_path}/{model_name}-label-pred-keywords-new-{trainingSession}.pkl", "rb") as f:
                pred_lable_keywords = pickle.load(f)
        
        predictions = pred_lable_keywords['predicitons']
        labels = pred_lable_keywords['labels']

        test_terms = set([lb for lbs in labels for lb in lbs])
        print(f"Session: {trainingSession} {model_name} {len(set([lb for label in labels for lb in label]))}")
        pred_terms = set([pr for pred in predictions for pr in pred])
        
        Num_train_terms.append(len(get_canonical_labels(train_terms)))
        Num_test_terms.append(len(get_canonical_labels(test_terms)))
        Num_same_terms.append(len(get_canonical_labels(train_terms).intersection(get_canonical_labels(test_terms))))
        Num_diff_terms.append(len(get_canonical_labels(test_terms).difference(get_canonical_labels(train_terms))))
        Num_Pred_terms.append(len(get_canonical_labels(pred_terms)))
        Num_pred_New_terms.append(len(get_canonical_labels(pred_terms).intersection(get_canonical_labels(test_terms).difference(get_canonical_labels(train_terms)))))
        Num_new_terms.append(len(pred_terms.difference(test_terms)))
    
    return Num_train_terms, Num_test_terms, Num_same_terms, Num_diff_terms, Num_Pred_terms, Num_pred_New_terms, Num_new_terms, New_terms

def get_new_words_ratio(file_path, trainingSession, withTag=True):
    newWordsRatios = []
    for model_name in models_name:
        if withTag:
            with open(f"{file_path}/{model_name}-label-pred-keywords-{trainingSession}.pkl", "rb") as f:
                pred_lable_keywords = pickle.load(f)
        else:
            with open(f"{file_path}/{model_name}-label-pred-keywords-new-{trainingSession}.pkl", "rb") as f:
                pred_lable_keywords = pickle.load(f)

        predictions = pred_lable_keywords['predicitons']
        labels = pred_lable_keywords['labels']

        num = 0
        for i, label in enumerate(labels):
            pred = predictions[i]
            new_pred = set(pred).difference(set(label))
            if len(list(new_pred)) != 0:
                num += 1
        
        newWordsRatios.append("{:.3f}".format(np.nan_to_num(num/len(labels))))
    
    return newWordsRatios

def get_predictions_df(file_path, data_path, trainingSession, withTag=True):
    with open(f"{data_path}/testset-{trainingSession}.pkl", "rb") as f:
        test_set = pickle.load(f)
    model_preds_df, tmp_df = pd.DataFrame({}), pd.DataFrame({})
    for i, model_name in enumerate(models_name):
        # print(f"{model_name}")
        if withTag:
            with open(f"{file_path}/{model_name}-label-pred-keywords-{trainingSession}.pkl", "rb") as f:
                pred_lable_keywords = pickle.load(f)
        else:
            with open(f"{file_path}/{model_name}-label-pred-keywords-new-{trainingSession}.pkl", "rb") as f:
                pred_lable_keywords = pickle.load(f)

        predictions = pred_lable_keywords['predicitons']
        labels = pred_lable_keywords['labels']
        
        # str_labels = []
        # for lb in labels:
        #     if len(lb) == 0:
        #         str_labels.append('')
        #     else:
        #         str_labels.append(str(lb))
        # print(len(test_set["sentence"].values.tolist()), len(test_set['keywords_0.1'].values.tolist()), len(predictions))
        
        new_df = pd.DataFrame({"sentences": test_set["sentence"].values.tolist(),
                               "keywords": [str(kws) for kws in test_set['keywords_0.1'].values.tolist()],
                               f"predictions({model_name})": predictions})
        if i == 0:
            tmp_df = pd.concat([tmp_df, new_df])
        elif i == 1:
            model_preds_df = tmp_df.merge(new_df, how='outer', on=['sentences', 'keywords'])
        else:
            model_preds_df = model_preds_df.merge(new_df, how='outer', on=['sentences', 'keywords'])

    return model_preds_df

def get_new_predictions_df(file_path, data_path, trainingSession, model, withTag=True):
    with open(f"{data_path}/testset-emb-{trainingSession}.pkl", "rb") as f:
        testset_emb = pickle.load(f)
    with open(f"{data_path}/testset-{trainingSession}.pkl", "rb") as f:
        test_set = pickle.load(f)

    model_preds_df, tmp_df = pd.DataFrame({}), pd.DataFrame({})
    for j, model_name in enumerate(models_name):
        if withTag:
            with open(f"{file_path}/{model_name}-label-pred-keywords-{trainingSession}.pkl", "rb") as f:
                pred_lable_keywords = pickle.load(f)
        else:
            with open(f"{file_path}/{model_name}-label-pred-keywords-new-{trainingSession}.pkl", "rb") as f:
                pred_lable_keywords = pickle.load(f)
        predictions = pred_lable_keywords['predicitons']
        labels = pred_lable_keywords['labels']
        new_keywords = []
        
        for i, pred in enumerate(predictions):
            label = labels[i]
            keyword = difference(pred, label)
            if keyword:
                new_keywords.append((i, keyword))

        num, sents, originals, news = filterAndCosineSim(new_keywords, testset_emb, test_set)
        new_df = pd.DataFrame({'sentences': sents, 'keywords': originals, f"newKeywords({model_name})": news}).reset_index(drop=True)
        print(f"\t{model_name}: {num / len(predictions):.3f}")

        if j == 0:
            tmp_df = pd.concat([tmp_df, new_df])
        elif j == 1:
            model_preds_df = tmp_df.merge(new_df, how='outer', on=['sentences', 'keywords'])
        else:
            # print(new_df.head())
            model_preds_df = model_preds_df.merge(new_df, how='outer', on=['sentences', 'keywords'])
    
    return model_preds_df

# Fill in the missing blanks
def data_aligned(pred_lable_keywords, from_pred_lable_keywords):
    predictions = pred_lable_keywords['predicitons']
    labels = pred_lable_keywords['labels']

    from_predictions = from_pred_lable_keywords['predicitons']
    from_labels = from_pred_lable_keywords['labels']

    if len(labels) == len(from_labels):
        print(len(labels), len(from_labels))
        print("nothing changed")
    else:
        print(labels[:10])
        print(from_labels[:100])
        print(from_predictions[:10])
        print(len(labels), len(from_labels))
        num = 0
        for i, label in enumerate(labels):
            if len(list(set(from_labels[i]) - set(label))) != 0:
                num += 1
                from_labels.insert(i, [label])
        print(num)
        print(len(labels), len(from_labels))
        # pred_cats = pred_lable_keywords['preds_cats']
        # labels_cats = pred_lable_keywords['labels_cats']

        # from_pred_cats = from_pred_lable_keywords['preds_cats']
        # from_labels_cats = from_pred_lable_keywords['labels_cats']
    return from_pred_lable_keywords

def difference(lst1, lst2):
    lst3 = [value for value in lst1 if value not in lst2]
    return lst3

def newPredsCounter(models_name):
    counter = []
    for preds in model_new_preds_df[f'newKeywords({models_name})'].values.tolist():
        if isinstance(preds, list):
            for pred in preds:
                counter.append(pred[1])
    return Counter(counter)

models_name = ['Bert-base-cased', 'Roberta', 'BiLSTM-CRF', 'CNN-CRF', 'CRF']

directory_file = '/Users/revekkakyriakoglou/Documents/paris_8/Projects/KeywordExtraction/'
data_file = directory_file + 'data/finalData/'
save_file = directory_file + 'models/results/'
cortext_dir = directory_file + 'data/originalData/'

data_path = "/content/drive/MyDrive/COURSE/Intern/Final_Experiments_13_09/All Datasets"
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

with open(directory_file + "data/intermediateData/0113_enriched_cat_web.pkl", "rb") as f:
    enrich_webset = pickle.load(f)
with open(directory_file + "data/intermediateData/0113_enriched_cat_ap.pkl", "rb") as f:
    enrich_apset = pickle.load(f)

gold_dict = enrich_webset | enrich_apset

for trainingSession in range(1,5):
    print(f"{trainingSession}:")
    recalls, precisions, f1scores = get_report(save_file, trainingSession, False)

    print(pd.DataFrame({"Model": models_name, "Recall": recalls, "Precision": precisions, "F1score": f1scores}).set_index("Model"))

for trainingSession in range(1,5):
    # newWordsRatios = get_new_words_ratio(file_path+str(trainingSession), trainingSession)
    Num_train_terms, Num_test_terms, Num_same_terms, Num_diff_terms, Num_Pred_terms, Num_pred_New_terms, Num_new_terms, New_terms = get_new_terms_amount(save_file, trainingSession, False)
    print(pd.DataFrame({"Model": models_name,
    "Recall": recalls,
    "Precision": precisions,
    "F1score": f1scores,
    "# Train Terms": Num_train_terms,
    '# Test Terms': Num_test_terms,
    '# Same Terms': Num_same_terms,
    '# Difference Terms (only in testing)': Num_diff_terms,
    '# Predictions': Num_Pred_terms,
    '# New terms (in the expert dictionary)': Num_pred_New_terms,
    '# New terms (out of expert dictionary)': Num_new_terms}).set_index("Model"))



