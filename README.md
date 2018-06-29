### DataCastle 基于人工智能的分子药物筛选 1-th place solution

[题目链接](http://www.dcjingsai.com/common/cmpt/%E5%9F%BA%E4%BA%8E%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E7%9A%84%E8%8D%AF%E7%89%A9%E5%88%86%E5%AD%90%E7%AD%9B%E9%80%89_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)

BeScientific

A榜第四: 1.22258  
B榜第一: 1.32833

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

所以，每个序列产生三个句子，用所有序列产生的句子训练出词向量。某蛋白质Sequence的w2v特征取为其三个句子中所有词的词向量的和。（发现这题取和比取平均好）
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

##### 2.4 蛋白质的统计特征 (make_protein_stat.py)

蛋白质Sequence中氨基酸个数（Sequence长度）, 20个氨基酸在Sequence中出现次数，出现频率
一共41个特征：protein_stat_0, protein_stat_1, ...., protein_stat_40

另外，还有个特征：protein_count: protein_id在训练集和测试集的出现次数，实现上和molecule_count发生了同样的错误（这个特征是在lgb.py里加的）

##### 2.5 Stacking特征 (make_stacking_feat.py, make_more_stacking.py)

关于Stacking: 训练集分为五折，各折训练集的该特征通过在其余四折数据上训练ridge得到，预测集的特征为5个ridge对预测集的预测平均值。

ridge_cat特征: 将molecule_id one hot (不是直接One Hot, molecule_id出现次数等于一的，归为了一类)，训练ridge

ridge_tfidf特征: 以protein_id的Sequence的tfidf作为特征，训练ridge

ridge_cat_tfidf特征：以one hot的molecule_id和tfidf作为特征，训练ridge

ridge_fp特征: 以fingerprint为特征，训练ridge

ridge_w2v特征：以w2v为特征，训练ridge

ridge_all: 以one hot的molecule_id和tfidf和df_molecule.csv中分子的物化属性（缺失值用均值填充）作为特征，训练ridge

##### 2.6 df_molecule.csv给出的分子的物化属性特征

#### 其它
这个视频 https://www.youtube.com/watch?v=LgLcfZjNF44 由些帮助
