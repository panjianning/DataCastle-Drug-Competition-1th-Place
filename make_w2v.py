import pandas as pd
import numpy as np
import re

from gensim.models import Word2Vec

from common import read_csv, timer

with timer("Load data"):
    df_protein_train = read_csv("df_protein_train.csv")
    df_protein_test = read_csv("df_protein_test.csv")

df_protein = pd.concat([df_protein_train,df_protein_test])
df_protein.Sequence = df_protein.Sequence.apply(lambda x: x.upper())

window_size = 3
embed_size = 128
w2v_col = [f'w2v_{i}' for i in range(embed_size)]
out_file = f'./input/temp/df_w2v_ws{window_size}.csv'

with timer("Tokenize sequence"):
    all_texts = []
    for seq in df_protein.Sequence:
        for shift in range(0,window_size):
            all_texts.append([word for word in re.findall(r'.{'+str(window_size)+'}',seq[shift:])])

with timer("Fit Word2Vec"):
    model = Word2Vec(all_texts,size=embed_size,window=4,min_count=1,negative=3,
                     sg=1,sample=0.001,hs=1,workers=4,iter=15)

with timer("Make sum_w2v feature"):
    w2v_feat = []
    i = 0
    while i <= len(all_texts)-window_size:
        sum_w2v = np.zeros(shape=(embed_size,))
        for j in range(i,i+window_size):
            for word in all_texts[j]:
                sum_w2v += model[word]
        w2v_feat.append(sum_w2v)
        i = i+window_size

with timer(f"Save file to {out_file}"):
    w2v_feat = np.vstack(w2v_feat)
    df_w2v = pd.DataFrame(w2v_feat,columns=w2v_col)
    df_w2v["Protein_ID"] = df_protein["Protein_ID"].values
    df_w2v[['Protein_ID']+w2v_col].to_csv(out_file,index=False)