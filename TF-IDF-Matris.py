###################################################
#Content Based Recommendation (İçerik Temelli Tavsiye)
###################################################
import numpy as np
##################################################
#film overview'larına göre tavsiye geliştirme
##################################################

# 1- TF-IDF Matrisinin oluşturulması
# 2- Cosine Similarity Matrisinin Oluşturulması
# 3- Benzerliklere Göre Önerilerin Yapılması
# 4- Çalışma Scriptinin Hazırlanması

####################################
# 1- TF-IDF MATRISININ OLUŞTURULMASI
#####################################

import pandas as pd
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.width",500)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# CSV dosyasını okuyun
df = pd.read_csv(r"datasets/movies_metadata.csv", low_memory=False)

# TfidfVectorizer nesnesini oluşturun
tfidf = TfidfVectorizer(stop_words="english")

# "overview" sütunundaki NaN değerleri boş bir dize ile doldurun
df["overview"] = df["overview"].fillna(" ")

# "overview" sütununu TF-IDF matrisine dönüştürün
tfidf_matrix = tfidf.fit_transform(df["overview"])

# TF-IDF matrisini kullanın
feature_names = tfidf.get_feature_names_out() # Kelime listesi

##########################################################
# Cosine Similarity Matrisi
##########################################################

cosine_sim = cosine_similarity(tfidf_matrix,tfidf_matrix)
cosine_sim.shape

#########################################################
# Benzerliklere Göre Öneri Yapma
########################################################

indices = pd.Series(df.index, index = df["title"] )

indices.value_counts() #çoklama var bben en son çekilen filmi istiyorun

indices = indices[~indices.index.duplicated(keep="last")]

indices["Cinderella"] # çoklamalar kalktı

movie_index = indices["Cinderella"]
similarity_score = pd.DataFrame(cosine_sim[movie_index],
                                columns = ["score"])

movie_indices = similarity_score.sort_values("score",ascending=False)

df["title"].iloc[movie_indices]
