###############################################################
# Item-Based Collaborative Filtering
###############################################################

#Adım 1 : Veri Setini Hazırlama
#Adım 2 : User Movie Df inin oluşturulması
#Adım 3 : Item-Based film önerilerinin yapılması
#Adım 4 : Çalışma Scriptinin Hazırlanması

###################################################################
# Adım 1 :  Veri Setinin Hazırlanması
##################################################################

import pandas as pd
pd.set_option("display.max_columns",500)

movie = pd.read_csv(r"datasets/movie.csv")
rating = pd.read_csv(r"datasets/rating.csv")
df = movie.merge(rating, how = "left", on = "movieId")

df.head()

###################################################################
# Adım 2 : User Movie Df 'nin Oluşturulması
###################################################################

# Her bir film için veri çerçevesindeki tekrar sayısını hesaplayın
rating_counts = pd.DataFrame(df["title"].value_counts()).reset_index()
rating_counts.columns = ["title", "count"]

# 1000 veya daha az inceleme alan nadir filmleri bulun
rare_movies = rating_counts[rating_counts["count"] <= 1000]["title"]

# Nadir filmleri hariç tutarak yeni bir veri çerçevesi oluşturun
common_movies = df[~df["title"].isin(rare_movies)]

# Her kullanıcının filmlerle ilgili derecelendirmelerini içeren bir çapraz tabloyu oluşturun
user_movie_df = common_movies.pivot_table(index="userId", columns="title", values="rating")

# İlk birkaç satırı görüntüle
user_movie_df.head()
#########################################################################
# Adım 3 : Item-Based Film Önerilerinin Yapılması
#########################################################################

movie_name = "Ocean's Twelve (2004)"
movie_name = user_movie_df[movie_name]

user_movie_df.corrwith(movie_name).sort_values(ascending = False).head()

def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]
check_film("Insomnia", user_movie_df)