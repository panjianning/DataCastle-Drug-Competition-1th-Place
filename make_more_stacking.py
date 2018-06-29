import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import gc

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from common import timer, read_csv, ItemSelector

fp_col = [f'fp_{i}' for i in range(167)]
w2v_col = [f'w2v_{i}' for i in range(128)]

with timer("Load data"):
    df_affinity_train = read_csv("df_affinity_train.csv")
    df_molecule = read_csv("df_molecule_stat.csv", "./input/temp/")
    df_protein_w2v = read_csv("df_w2v_ws3.csv", "./input/temp/")
    df_affinity_test = read_csv("df_affinity_test_toBePredicted.csv")

df_train = df_affinity_train.merge(df_molecule)
df_train = df_train.merge(df_protein_w2v)

df_test = df_affinity_test.merge(df_molecule)
df_test = df_test.merge(df_protein_w2v)

test = df_test
all_protein_id = df_train.Protein_ID.unique()

kfold = KFold(n_splits=5,shuffle=True,random_state=2018)

df_data = []

meta_feat = ['ridge_fp','ridge_w2v']

for feat in meta_feat:
    test[feat] = 0

for train_idx, valid_idx in kfold.split(all_protein_id):
    train_protein_id = all_protein_id[train_idx]
    valid_protein_id = all_protein_id[valid_idx]
    train = df_train[df_train.Protein_ID.isin(train_protein_id)]
    valid = df_train[df_train.Protein_ID.isin(valid_protein_id)]
    
    fp_feat = ItemSelector(key=fp_col)
    w2v_feat = ItemSelector(key=w2v_col)

    fp_train = fp_feat.fit_transform(train)
    fp_valid = fp_feat.transform(valid)
    fp_test = fp_feat.transform(test)
    
    w2v_train = w2v_feat.fit_transform(train)
    w2v_valid = w2v_feat.transform(valid)
    w2v_test = w2v_feat.transform(test)

    y_train = train.Ki.values
    y_valid = valid.Ki.values
    
    model_ridge_fp = Ridge(alpha=1,random_state=2018)
    model_ridge_w2v = Ridge(alpha=1,random_state=2018)

    with timer("Fit ridge (fp)"):
        model_ridge_fp.fit(fp_train,y_train)

    with timer("Fit ridge (w2v)"):
        model_ridge_w2v.fit(w2v_train,y_train)

    test['ridge_fp'] += model_ridge_fp.predict(fp_test)
    valid['ridge_fp'] = model_ridge_fp.predict(fp_valid)  

    test['ridge_w2v'] += model_ridge_w2v.predict(w2v_test)
    valid['ridge_w2v'] = model_ridge_w2v.predict(w2v_valid)    

    df_data.append(valid.copy())    
    
for feat in meta_feat:
    test[feat] = test[feat]/5

out_col = ['Protein_ID','Molecule_ID']+meta_feat

df_out = pd.concat(df_data,axis=0)

with timer(f"Save file"):
    df_out[out_col].to_csv('./input/temp/df_more_meta_train.csv',index=False)
    test[out_col].to_csv('./input/temp/df_more_meta_test.csv',index=False)