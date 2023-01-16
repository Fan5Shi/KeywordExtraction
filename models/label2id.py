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