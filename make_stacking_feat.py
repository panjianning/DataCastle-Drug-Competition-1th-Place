import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split,KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import gc

from collections import Counter

import os
from scipy.sparse import hstack,csr_matrix
import warnings
warnings.filterwarnings('ignore')

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

with timer("Load data"):
    df_protein_train = read_csv("df_protein_train.csv")
    df_affinity_train = read_csv("df_affinity_train.csv")
    df_molecule = read_csv("df_molecule.csv")

    df_protein_test = read_csv("df_protein_test.csv")
    df_affinity_test_toBePredicted = read_csv("df_affinity_test_toBePredicted.csv")

    df_protein = pd.concat([df_protein_train,df_protein_test])
    df_protein.sort_values("Protein_ID",inplace=True)

molecule_field = [
        'cyp_3a4', 'cyp_2c9', 'cyp_2d6',
       'ames_toxicity', 'fathead_minnow_toxicity',
       'tetrahymena_pyriformis_toxicity', 'honey_bee', 'cell_permeability',
       'logP', 'renal_organic_cation_transporter', 'CLtotal', 'hia',
       'biodegradation', 'Vdd', 'p_glycoprotein_inhibition', 'NOAEL',
       'solubility', 'bbb']

def molecule_fillna(df,select_col):
    scaler = MinMaxScaler(feature_range=(0,1))
    for col in select_col:
        df[col] = scaler.fit_transform(df[[col]].fillna(df[col].mean()))
    return df

with timer("Molecule fillna"):
    df_molecule = molecule_fillna(df_molecule,molecule_field)


df_train = df_affinity_train.merge(df_protein)
df_train = df_train.merge(df_molecule)

df_test = df_affinity_test_toBePredicted.merge(df_protein)
df_test = df_test.merge(df_molecule)

test = df_test

all_protein_id = df_train.Protein_ID.unique()

kfold = KFold(n_splits=5,shuffle=True,random_state=2018)

df_data = []

meta_feat = ['ridge_cat','ridge_tfidf','ridge_cat_tfidf','ridge_all']

for feat in meta_feat:
    test[feat] = 0

for train_idx, valid_idx in kfold.split(all_protein_id):

    train_protein_id = all_protein_id[train_idx]
    valid_protein_id = all_protein_id[valid_idx]

    train = df_train[df_train.Protein_ID.isin(train_protein_id)]
    valid = df_train[df_train.Protein_ID.isin(valid_protein_id)]

    print(train.shape, valid.shape)

    df_tmp = pd.concat([df_train,df_test]).groupby("Molecule_ID",as_index=False).Protein_ID.agg(
        {"how_many":"count"}).sort_values('how_many',ascending=False)

    molecule_id = list(df_tmp[df_tmp.how_many!=1].Molecule_ID.values)
    molecule_id.append(99999)
    molecule_id = np.array(molecule_id).reshape(-1,1)

    train_molecule_id = train.Molecule_ID.copy()
    train_molecule_id[~train_molecule_id.isin(molecule_id.ravel())]=99999

    valid_molecule_id = valid.Molecule_ID.copy()
    valid_molecule_id[~valid_molecule_id.isin(molecule_id.ravel())]=99999

    test_molecule_id = test.Molecule_ID.copy()
    test_molecule_id[~test_molecule_id.isin(molecule_id.ravel())]=99999

    encoder = OneHotEncoder()
    encoder.fit(molecule_id)
    molecule_cat_feat_train = encoder.transform(train_molecule_id.values.reshape(-1,1))
    molecule_cat_feat_valid = encoder.transform(valid_molecule_id.values.reshape(-1,1))
    molecule_cat_feat_test = encoder.transform(test_molecule_id.values.reshape(-1,1))
    
    molecule_feat_train = csr_matrix(train[molecule_field].values)
    print("make molecule_feat_valid")
    molecule_feat_valid = csr_matrix(valid[molecule_field].values)
    print("make molecule_feat_test")
    molecule_feat_test = csr_matrix(test[molecule_field].values)

    tfidf_vec = TfidfVectorizer(max_features=100000, analyzer='char', ngram_range=(1,3))
    tfidf_vec.fit_transform(df_protein.Sequence)

    tfidf_feat_train = tfidf_vec.transform(train.Sequence)
    tfidf_feat_valid = tfidf_vec.transform(valid.Sequence)
    tfidf_feat_test = tfidf_vec.transform(test.Sequence)

    cat_tfidf_train = hstack([tfidf_feat_train,molecule_cat_feat_train],format="csr")
    cat_tfidf_valid = hstack([tfidf_feat_valid,molecule_cat_feat_valid],format="csr")
    cat_tfidf_test = hstack([tfidf_feat_test,molecule_cat_feat_test],format="csr")

    x_train = hstack([tfidf_feat_train,molecule_feat_train,molecule_cat_feat_train],format="csr")
    x_valid = hstack([tfidf_feat_valid,molecule_feat_valid,molecule_cat_feat_valid],format="csr")
    x_test = hstack([tfidf_feat_test,molecule_feat_test,molecule_cat_feat_test],format="csr")
    
    y_train = train.Ki.values
    y_valid = valid.Ki.values

    with timer('Fit Ridge (cat)'):
        ridge = Ridge(alpha=1,random_state=2018)
        ridge.fit(molecule_cat_feat_train,y_train)
    prediction = ridge.predict(molecule_cat_feat_valid)
    test['ridge_cat'] += ridge.predict(molecule_cat_feat_test)
    valid['ridge_cat'] = prediction
    
    with timer('Fit Ridge (tfidf)'):
        ridge = Ridge(alpha=1,random_state=2018)
        ridge.fit(tfidf_feat_train,y_train)
    prediction = ridge.predict(tfidf_feat_valid)
    test['ridge_tfidf'] += ridge.predict(tfidf_feat_test)
    valid['ridge_tfidf'] = prediction

    with timer('Fit Ridge (cat_tfidf)'):
        ridge = Ridge(alpha=1,random_state=2018)
        ridge.fit(cat_tfidf_train,y_train)
    prediction = ridge.predict(cat_tfidf_valid)
    test['ridge_cat_tfidf'] += ridge.predict(cat_tfidf_test)
    valid['ridge_cat_tfidf'] = prediction    

    with timer("Fit Ridge (all)"):
        ridge = Ridge(alpha=1,random_state=2018)
        ridge.fit(x_train,y_train)
    prediction = ridge.predict(x_valid)
    test['ridge_all'] += ridge.predict(x_test)
    valid['ridge_all'] = prediction    

    df_data.append(valid.copy())
    
for feat in meta_feat:
    test[feat] = test[feat]/5

out_col = ['Protein_ID','Molecule_ID']+meta_feat

df_meta_train = pd.concat(df_data,axis=0)
df_meta_train[out_col].to_csv('./input/temp/df_meta_train.csv',index=False)
test[out_col].to_csv('./input/temp/df_meta_test.csv',index=False)