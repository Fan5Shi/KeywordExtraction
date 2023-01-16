import pickle

directory_file = '/workspaces/KeywordExtraction/'
saved_file = directory_file + 'data/finalData/'
trainingSession = 1

with open(saved_file + f"trainset-{trainingSession}.pkl", "rb") as f:
    train_set = pickle.load(f)
with open(saved_file + f"testset-{trainingSession}.pkl", "rb") as f:
    test_set = pickle.load(f)