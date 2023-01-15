import pandas as pd
import numpy as np
import pickle

# from dataset_processing import 

def generate_indices(test_df, ref_terms, kw_col):
    indices = []
    for i in range(len(test_df)):
        kws = test_df[kw_col].iloc[i]
        canbetrain = False
        for kw in kws:
            if kw not in ref_terms:
                canbetrain = True
                break
        if not canbetrain:
            indices.append(i)
    
    return indices

def SetSampler(datasetdf, sample_size, kw_col):
    keywords = set(kw for kws in datasetdf[kw_col].values.tolist() for kw in kws)

    keywords2blocks = {kw:[] for kw in keywords}
    blocks2keywords = {i:datasetdf[kw_col].loc[i] for i in datasetdf.index}

    # Update keywords2blocks
    for k in keywords2blocks.keys():
        for kb, vb in blocks2keywords.items():
            if k in vb:
                keywords2blocks[k].append(kb)
                
    keyword_list, block_list = [], []
    for k,v in keywords2blocks.items():
        keyword_list.extend([k]*len(v))
        block_list.extend(v)

    reductiondf = pd.DataFrame({kw_col:keyword_list, 'block': block_list})

    sample_size = int(len(datasetdf) * sample_size)
    sample = reductiondf.groupby(kw_col).sample(1, random_state=1)

    sample = sample.append(
        reductiondf[~reductiondf.index.isin(sample.index)] # only rows that have not been selected
        .sample(n=sample_size-sample.shape[0]) # sample more rows as needed
    ).sort_index()
    
    return list(set(sample['block'].values.tolist()))

'''
    Function to build our final dataset
    kw_col: the keyword column we want to use, for now there are two
            choices, "keywords_0.1" and "keywords_0.23"
    label: from 1 to 4, represents the four different methods we want
            to build the dataset
'''
def FinalDatasetBuilder(df, kw_col="keywords_0.1", label=1):
    # remove all the blocks that don't have keywords
    datasetdf = df[df[kw_col].map(len) != 0]
    
    if label == 1:
        train_setdf = datasetdf[datasetdf['source'] == 'web']
        test_setdf = datasetdf[datasetdf['source'] == 'ap']
        
    elif label == 2:
        train_setdf = datasetdf[datasetdf['source'] == 'web']
        test_setdf = datasetdf[datasetdf['source'] == 'ap']
        
        '''
        0.659040590405904 0.34095940959409593
        The size of the training set is: 10697
        The size of the testing set is: 33279
        '''
        np.random.seed(442)
        all_terms = set([t for tt in train_setdf[kw_col].values.tolist() for t in tt])
        train_terms = np.random.choice(list(all_terms), size=int(len(all_terms)*0.8), replace=False)
#         test_terms = all_terms.difference(train_terms)
        
        train_indices = generate_indices(train_setdf, train_terms, kw_col)
#         test_indices = generate_indices(train_setdf, test_terms, kw_col)
        
        web_train_setdf = train_setdf.iloc[train_indices]
        web_test_setdf = train_setdf.drop(web_train_setdf.index)
        test_setdf = pd.concat([test_setdf, web_test_setdf])
        print(len(train_indices)/len(train_setdf), len(web_test_setdf.index)/len(train_setdf))
        train_setdf = web_train_setdf
        
    elif label == 3:
        web_setdf = datasetdf[datasetdf['source'] == 'web']
        ap_setdf = datasetdf[datasetdf['source'] == 'ap']
        print(len(web_setdf), len(ap_setdf))
        
        np.random.seed(442)
        all_terms = set([t for tt in web_setdf[kw_col].values.tolist() for t in tt])
        train_terms = np.random.choice(list(all_terms), size=int(len(all_terms)*0.7), replace=False)
#         test_terms = all_terms.difference(train_terms)
        
        web_train_indices = generate_indices(web_setdf, train_terms, kw_col)
        ap_train_indices = generate_indices(ap_setdf, train_terms, kw_col)
        print(len(web_train_indices), len(web_train_indices)/len(web_setdf))
        print(len(ap_train_indices), len(ap_train_indices)/len(ap_setdf))
        
        # sampling size is computed by: 25% / len(ap_train_indices)/len(ap_setdf)
        ap_train_indices = SetSampler(ap_setdf.iloc[ap_train_indices], 0.4, kw_col)
        print(len(ap_train_indices), len(ap_train_indices)/len(ap_setdf))
        
        web_train_setdf = web_setdf.iloc[web_train_indices]
        ap_train_setdf = ap_setdf.loc[ap_train_indices]
        web_test_setdf = web_setdf.drop(web_train_setdf.index)
        ap_test_setdf = ap_setdf.drop(ap_train_setdf.index)
        test_setdf = pd.concat([ap_test_setdf, web_test_setdf])
        train_setdf = pd.concat([web_train_setdf, ap_train_setdf])
        print(len(train_setdf), len(test_setdf))
        
    elif label == 4:
        web_setdf = datasetdf[datasetdf['source'] == 'web']
        ap_setdf = datasetdf[datasetdf['source'] == 'ap']
        print(len(web_setdf), len(ap_setdf))
        
        np.random.seed(442)
        all_terms = set([t for tt in web_setdf[kw_col].values.tolist() for t in tt])
        train_terms = np.random.choice(list(all_terms), size=int(len(all_terms)*0.5), replace=False)
        test_terms = all_terms.difference(train_terms)
        
        web_train_indices = generate_indices(web_setdf, train_terms, kw_col)
        ap_train_indices = generate_indices(ap_setdf, test_terms, kw_col)
        print(len(web_train_indices), len(web_train_indices)/len(web_setdf))
        print(len(ap_train_indices), len(ap_train_indices)/len(ap_setdf))
        
        web_train_setdf = web_setdf.iloc[web_train_indices]
        ap_train_setdf = ap_setdf.iloc[ap_train_indices]
        web_test_setdf = web_setdf.drop(web_train_setdf.index)
        ap_test_setdf = ap_setdf.drop(ap_train_setdf.index)
        test_setdf = pd.concat([ap_test_setdf, web_test_setdf])
        train_setdf = pd.concat([web_train_setdf, ap_train_setdf])
        print(len(train_setdf), len(test_setdf))
    
    else:
        # Extra test, collection of blocks that don't have any keywords
        test_setdf = df[df[kw_col].map(len) == 0]
        print(f"The size of the testing set is: {len(test_setdf)}")
        return pd.DataFrame({}), test_setdf
    
    print(f"The size of the training set is: {len(train_setdf)}")
    print(f"The size of the testing set is: {len(test_setdf)}")
    print("------------------------")
    return train_setdf, test_setdf

def main():
    directory_file = '/workspaces/KeywordExtraction/'
    saved_file = directory_file + 'data/finalData/'

    with open(directory_file + "data/intermediateData/tagged_dataset_web_01_13.pkl", "rb") as f:
        sem_webset = pickle.load(f)
    with open(directory_file + "data/intermediateData/tagged_dataset_ap_01_13.pkl", "rb") as f:
        sem_apset = pickle.load(f)
    
    sem_webset['source'] = 'web'
    sem_apset['source'] = 'ap'

    datasetdf_sem = pd.concat([sem_webset, sem_apset])

    print(len(sem_webset), len(sem_apset))
    print(len(datasetdf_sem))

    for label in range(5):
        train_set, test_set = FinalDatasetBuilder(datasetdf_sem, kw_col='keywords', label=label)

        train_set.to_pickle(saved_file + f"trainset-{label}.pkl")
        test_set.to_pickle(saved_file + f"testset-{label}.pkl")
        print(len(train_set), len(test_set))

if __name__ == "__main__":
    main()