import pickle

with open("/workspaces/KeywordExtraction/data/finalData/trainset-1.pkl", "rb") as f:
    test = pickle.load(f)

test.head(5)