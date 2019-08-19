
import pandas as pd
import numpy as np
import re

from konlpy.tag import Okt
okt = Okt()

from gensim.models import Word2Vec

from sklearn.decomposition import PCA

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
path = '/usr/share/fonts/NanumFont/NanumMyeongjo.ttf'
font_name = fm.FontProperties(fname=path, size=50).get_name()
print(font_name)
plt.rc('font', family=font_name)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
# brand 리스트 선별
brand_list = df['N_BRAND_NAME'].value_counts().head(40).index.to_list()
sample_list = brand_list[:15] # 샘플
# 선별된 브랜드 리스트 데이터만 가져오기
df = df[df['N_BRAND_NAME'].apply(lambda x : x in sample_list)]
# 정확도를 높히기 위한 GOODS_NAME에 중복되는 브랜드 네임 제거
len_ = len (df)
for i in range(len_):
    word_tokens = word_tokenize(df['GOODS_NAME'].values[i])
    result = []
    for w in word_tokens:
        if w not in sample_list:
            result.append(w)
    df['GOODS_NAME'].values[i] = ' '.join(result)
# 데이터 클리닝
def cleanText(readData):
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)
    return text
df['GOODS_NAME'] = df['GOODS_NAME'].apply(cleanText)
# 데이터 클리닝2
def spacing_okt(wrongSentence):
    tagged = okt.pos(wrongSentence)
    corrected = ""
    for i in tagged:
        if i[1][0] in "JEXSO":
            corrected += i[0]
        else:
            corrected += " "+i[0]
    try: # 수정부분
        if corrected[0] == " ": 
            corrected = corrected[1:]
    except:
        pass
    return corrected
df['GOODS_NAME'] = df['GOODS_NAME'].apply(spacing_okt)
# 문장 만들기
df['W2V'] = df['N_BRAND_NAME'] + ' ' + df['GOODS_NAME']
# 중복 제거
df['W2V'] = df['W2V'].drop_duplicates()
df = df.dropna(axis = 0)
# 문장 리스트 구성
df['W2V'] = df['W2V'].apply(lambda x:x.split(' '))
# 문장 리스트화
sentences = [x for x in df['W2V']]
# 문장을 이용하여 단어와 벡터를 생성
model = Word2Vec(sentences, size=300, window=3, min_count=1, workers=1)
# 단어벡터를 구한다.
word_vectors = model.wv
vocabs = word_vectors.vocab.keys()
word_vectors_list = [word_vectors[v] for v in vocabs]
# 차원 축소
pca = PCA(n_components=2)
xys = pca.fit_transform(word_vectors_list)
xs = xys[:,0]
ys = xys[:,1]
# 그래프 그리기
list_n = [list(vocabs).index(brand) for brand in sample_list] 

plt.figure(figsize=(20,5))
for i in list_n:
    plt.scatter(xs[i], ys[i], marker = '*', color = 'r')
    plt.annotate(list(vocabs)[i], xy=(xs[i], ys[i]),fontsize = 13)
plt.show()
# 클러스터링
k_values = [[xs[i],ys[i]] for i in list_n]
k_df = pd.DataFrame(k_values).rename(columns={0:'X',1:'Y'})
kmeans = KMeans(n_clusters = 5).fit(k_values) 
k_df['cluster'] = kmeans.labels_
# 클러스터링 그래프 그리기
sns.lmplot('X','Y', data = k_df, fit_reg = False, scatter_kws = {"s":20}, hue = "cluster")
for i in list_n:
    plt.annotate(list(vocabs)[i], xy=(xs[i], ys[i]),fontsize = 10)
plt.show()
# 클러스터링 평가
k_df['score_samples'] = silhouette_samples(k_df, k_df['cluster'])
k_df['score_samples'].mean()
### 여러개의 클러스터링 갯수를 List로 입력 받아 각각의 실루엣 계수를 면적으로 시각화한 함수 작성
def visualize_silhouette(cluster_lists, X_features): 
    
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import math
    
    n_cols = len(cluster_lists)
    
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    
    for ind, n_cluster in enumerate(cluster_lists):
        
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(X_features)
        
        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)
        
        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster) + 'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort()
            
            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values,facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")
visualize_silhouette([2,3,4,5,6,7,8,9], k_df)

