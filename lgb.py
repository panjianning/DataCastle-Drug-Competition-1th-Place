import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
import gc

from collections import Counter
import os

import time
from contextlib import contextmanager
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')
    
def read_csv(fname, input_path="./input/", **kwargs):
    full_path = os.path.join(input_path,fname)
    return pd.read_csv(full_path,**kwargs)

w2v_col = [f'w2v_{i}' for i in range(128)]
fp_col = [f'fp_{i}' for i in range(167)]
mol_col = [
    "cyp_3a4", "cyp_2c9", "cyp_2d6","ames_toxicity",                       
    "fathead_minnow_toxicity","tetrahymena_pyriformis_toxicity",     
    "honey_bee", "cell_permeability",                   
    "logP", "renal_organic_cation_transporter",    
    "CLtotal", "hia",                                 
    "biodegradation","Vdd",                                 
    "p_glycoprotein_inhibition", "NOAEL",                               
    "solubility", "bbb"
    ]
mol_stat_col = ['molecule_count']
pro_stat_col = [f'protein_stat_{i}' for i in range(41)]+['protein_count']
stack_col = [
    'ridge_all', 'ridge_tfidf','ridge_cat',
    'ridge_fp','ridge_cat_tfidf','ridge_w2v'
    ]

with timer("Load data"):
    df_affinity_train = read_csv("df_affinity_train.csv")
    df_affinity_test = read_csv("df_affinity_test_toBePredicted.csv")
    df_molecule = read_csv("df_molecule.csv").drop(["Fingerprint"],axis=1)

    df_protein_stat = read_csv("df_protein_stat.csv", './input/temp')
    df_molecule_stat = read_csv("df_molecule_stat.csv", './input/temp')

    df_protein_w2v = read_csv("df_w2v_ws3.csv", './input/temp')
    
    df_meta_train = read_csv("df_meta_train.csv", './input/temp')
    df_meta_test = read_csv("df_meta_test.csv", './input/temp')

    df_more_meta_train = read_csv("df_more_meta_train.csv",'./input/temp')
    df_more_meta_test = read_csv("df_more_meta_test.csv",'./input/temp')    
    
    df_protein_count = pd.concat([df_affinity_train,df_affinity_test])
    df_protein_count = df_protein_count.groupby('Protein_ID',as_index=False).Ki.agg({'protein_count':'count'})
    
    df_protein_stat = df_protein_stat.merge(df_protein_count)

with timer("Merge data"):
    
    df_meta_train = df_meta_train.merge(df_more_meta_train,on=['Protein_ID','Molecule_ID'])
    df_meta_test = df_meta_test.merge(df_more_meta_test,on=['Protein_ID','Molecule_ID'])

    df_protein = df_protein_stat.merge(df_protein_w2v,on=['Protein_ID'])
    df_molecule = df_molecule.merge(df_molecule_stat,on=['Molecule_ID'])

    df_train = df_affinity_train.merge(df_meta_train,on=['Protein_ID','Molecule_ID'])
    df_train = df_train.merge(df_protein,on=['Protein_ID'])
    df_train = df_train.merge(df_molecule,on=['Molecule_ID'])

    df_test = df_affinity_test.merge(df_meta_test,on=['Protein_ID','Molecule_ID'])
    df_test = df_test.merge(df_protein,on=['Protein_ID'])
    df_test = df_test.merge(df_molecule,on=['Molecule_ID'])
    
    test = df_test

with timer('Check shape'):
    print(df_train.shape, df_affinity_train.shape)
    print(df_test.shape, df_affinity_test.shape)


with timer("Train valid split"):
    train_protein_id = df_train.Protein_ID.unique()
    # train_protein_id, valid_protein_id = train_test_split(protein_id,
    #                                                       test_size=0.0,shuffle=True,random_state=0)

    train = df_train[df_train.Protein_ID.isin(train_protein_id)]
    # valid = df_train[df_train.Protein_ID.isin(valid_protein_id)]
    
# with timer('Check shape'):
#     print(train.shape,valid.shape,test.shape)

select_col = mol_col + stack_col + pro_stat_col + mol_stat_col + w2v_col + fp_col

with timer('Make lgb data'):
    lgbtrain = lgb.Dataset(data=train[select_col].values, label=train["Ki"].values, feature_name=select_col)
    # lgbvalid = lgb.Dataset(data=valid[select_col].values, label=valid["Ki"].values, feature_name=select_col)
    
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': 'l2',
    "learning_rate": 0.005,
    "max_depth":-1,
    "num_leaves":200,
    "max_bin":250,
    "colsample_bytree": 0.5,
    'reg_lambda': 2.8,
    'min_data_in_leaf':20,
}

with timer('Train lgb model'):
    booster = lgb.train(lgb_params,
                        lgbtrain,
                        valid_sets=[lgbtrain],
                        verbose_eval=1,
                        num_boost_round=2500,
                        early_stopping_rounds=80)

with timer('Generate submission'):
    prediction = booster.predict(test[select_col].values,num_iteration=booster.best_iteration)
    test['Ki'] = prediction
    test[['Protein_ID','Molecule_ID','Ki']].to_csv('submission.csv',index=False)