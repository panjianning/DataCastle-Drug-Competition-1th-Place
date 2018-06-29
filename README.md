### DataCastle 基于人工智能的分子药物筛选 1-th place solution

#### 一．代码运行环境及依赖
Python 3.6 

Ubuntu 18.04

16G + 内存

需要用到的Python包: numpy, pandas, sklearn, lightgbm, gensim

#### 二．模型
LightGBM

#### 三. 特征
##### 2.1 word2vec特征 （make_w2v.py）
对蛋白质Sequence进行分词，每三个氨基酸视为一个词，一共有20^3 = 8000个词。

【Example】设序列为ABCDEFGHIJK, 设三个氨基酸一个词，该序列可以分为3个句子

i. [‘ABC’, ‘DEF’, ‘GHI’]

ii. [‘BCD’, ‘EFG’, ‘HIJ’]

iii. [‘CDE’, ‘FGH’, ‘IJK’]

所以，每个序列产生三个句子，用这些句子训练出词向量。某蛋白质Sequence的w2v特征取为其三个句子中所有词的词向量的和。（发现这题取和比取平均好）
在代码中，word embedding size为128, 所以此时产生了128个特征：

w2v_0, w2v_1, ...., w2v_127

##### 2.2 fingerprint特征 (make_molecule_stat.py)
只是单纯的把fingerprint split开，产生167个Binary feature 

##### 2.3 分子的统计特征 (make_molecule_stat.py)
molecule_id在训练集+测试集的出现次数：molecule_count

但实现上，我用了下面的代码去统计
<pre><code>
df_aff = pd.concat([df_aff_train,df_aff_test])
df_molecule_count = df_aff.groupby("Molecule_ID",as_index=False).Ki.agg({"molecule_count":"count"})
</code></pre>
concat时，df_aff中来自测试集的Ki是missing的，而count操作时不计missing values的，所以，实际上我算的是molecule_id在训练集的出现次数，而测试集的molecule_count全为0...
