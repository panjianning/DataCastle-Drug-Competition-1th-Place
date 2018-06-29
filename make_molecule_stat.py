import pandas as pd
import numpy as np

from common import timer,read_csv

fp_col = [f'fp_{i}' for i in range(167)]
out_col = ["molecule_count"]+fp_col
output_file = "./input/temp/df_molecule_stat.csv"

with timer("Load data"):
	df_molecule = read_csv("df_molecule.csv")
	df_aff_train = read_csv("df_affinity_train.csv")
	df_aff_test = read_csv("df_affinity_test_toBePredicted.csv")

df_aff = pd.concat([df_aff_train,df_aff_test])

with timer("Make molecule count feature"):
	df_molecule_count = df_aff.groupby("Molecule_ID",as_index=False).Ki.agg({"molecule_count":"count"})
	df_molecule = df_molecule.merge(df_molecule_count,on=["Molecule_ID"])

with timer("Parse fingerprint"):
    fingerprint = df_molecule.Fingerprint.apply(lambda x: np.array(x.split(', '))).values
    fingerprint = np.vstack(fingerprint).astype(np.uint8)
    df_fingerprint = pd.DataFrame(fingerprint,columns=fp_col,dtype=np.uint8)
    df_molecule = pd.concat([df_molecule,df_fingerprint],axis=1)
    del df_fingerprint, fingerprint

with timer(f"Save file to {output_file}"):
	df_molecule[['Molecule_ID']+out_col].to_csv(output_file,index=False)