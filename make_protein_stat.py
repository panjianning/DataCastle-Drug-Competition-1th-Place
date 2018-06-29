import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.pipeline import FeatureUnion,Pipeline,make_union,make_pipeline

from common import timer,read_csv,ItemSelector,TextStats

with timer("Load data"):
	df_protein_train = read_csv("df_protein_train.csv")
	df_protein_test = read_csv("df_protein_test.csv")

df_protein = pd.concat([df_protein_train,df_protein_test])
df_protein.Sequence = df_protein.Sequence.apply(lambda x: x.upper())

feature_union = make_union(
	make_pipeline(ItemSelector(key="Sequence"),CountVectorizer(analyzer='char',ngram_range=(1,1))),
	make_pipeline(ItemSelector(key="Sequence"),TfidfVectorizer(analyzer='char',ngram_range=(1,1),use_idf=False)),
	make_pipeline(ItemSelector(key="Sequence"),TextStats(), DictVectorizer())
	)

with timer("Fit feature_union"):
	feat = feature_union.fit_transform(df_protein)

out_col = [f'protein_stat_{i}' for i in range(feat.shape[1])]
output_file = "./input/temp/df_protein_stat.csv"

with timer(f"Save file to {output_file}"):
	df_out = pd.DataFrame(feat.todense(),columns=out_col)
	df_out['Protein_ID'] = df_protein.Protein_ID.values
	df_out[['Protein_ID']+out_col].to_csv(output_file,index=False)