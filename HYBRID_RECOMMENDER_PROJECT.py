
#############################################
# PROJE: Hybrid Recommender System
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recomennder yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapınız.

#############################################
# Görev 1: Verinin Hazırlanması
#############################################


# Adım 1: Movie ve Rating veri setlerini okutunuz.
# movieId, film adı ve filmin tür bilgilerini içeren veri seti

import pandas as pd



rating = pd.read_csv(r"datasets/rating.csv")


# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti

movie = pd.read_csv(r"datasets/movie.csv")

# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyiniz.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.

movie_rate = movie.merge(rating,how = "inner")


# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayınız.Toplam oy kullanılma sayısı 1000'un altında olan filmleri veri setinden çıkarınız.
# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.

rare_movies = pd.DataFrame(movie_rate["title"].value_counts()).reset_index()
rare_movies.columns = ["title","count"]
print(rare_movies)



# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz

rare_movies = rare_movies[rare_movies["count"] < 1000]["title"]

commen_movies = movie_rate[~movie_rate["title"].isin(rare_movies)]


# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.

commen_movies_df = commen_movies.pivot_table(index = "userId", columns = "title", values = "rating")

# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım

def create_user_movie_df():
    df = movie.merge(rating, how = "left", on = "movieId")
    rating_counts = pd.DataFrame(df["title"].value_counts()).reset_index()
    rating_counts.columns = ["title", "count"]

    # 1000 veya daha az inceleme alan nadir filmleri bulun
    rare_movies = rating_counts[rating_counts["count"] < 1000]["title"]

    # Nadir filmleri hariç tutarak yeni bir veri çerçevesi oluşturun
    common_movies = df[~df["title"].isin(rare_movies)]

    # Her kullanıcının filmlerle ilgili derecelendirmelerini içeren bir çapraz tabloyu oluşturun
    user_movie_df = common_movies.pivot_table(index="userId", columns="title", values="rating")


    return user_movie_df

#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Rastgele bir kullanıcı seçme
random_user = int(pd.Series(commen_movies_df.index).sample(1, random_state=45).values[0])

# Seçilen kullanıcının oy kullandığı filmleri içeren DataFrame'i oluşturma
random_user_df = commen_movies_df[commen_movies_df.index == random_user]

# Bu kullanıcının izlediği filmleri bir liste olarak alın
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# Elde edilen izlediği filmleri görüntüleme
print(movies_watched)


#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.

movies_watched_df = commen_movies_df[movies_watched]

# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
# Ve yeni bir df oluşturuyoruz.

user_movie_count = movies_watched_df.T.notnull().sum() #null olanlarda kullanıcı o filmi izlememiştir

user_movie_count = user_movie_count.reset_index()
user_movie_count.head()
user_movie_count.columns = ["userId", "movie_count"]

# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturunuz.

len(movies_watched) # sectigimiz random kullanıcının izledigi film sayısı
perc = len(movies_watched) * 60 / 100 # eşik degeri olacak film sayısını hesaplıyoruz.

users_same_movies = user_movie_count[user_movie_count["movie_count"] >= perc]["userId"]


#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.

final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)] # user_same_movies: benzerligi %60 üzerinde olan user'lar.
final_df.head()

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.


#corr_df[corr_df["user_id_1"] == random_user]

final_df.T.corr().unstack() # pivot hale getiriyoruz.
corr_df = final_df.T.corr().unstack().sort_values()




# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.

corr_df = pd.DataFrame(corr_df, columns=["corr"]) # korelasyon degerlerini yeni bir sutun olusturarak bu sutuna yerleştiriyoruz
corr_df.index.names = ['user_id_1', 'user_id_2'] #  user sutunlarını isimlendiriyoruz.
corr_df.head()
corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False) # secilen random kulalnıcı ile 0.65 degerin üzerinde korelasyona sahip kullanıcıların sıralanmıs hali.
top_users.shape

# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner') # rating veri setinde "userId", "movieId", "rating" değişkenlerini alıyoruz.
# inner: ortak olanları getirir.
top_users_ratings.tail() # sectigimiz random izleyicinin kendisine benzer -> corr degeri 0.65 üstü olan kullanıcılar ile izledigi filmlere verdiği rating degerleri.
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user] # veriden random user'ın kendi id'sini cıkartıyoruz
top_users_ratings["userId"].unique() #kendisine benzer izleyicilerin user'id'leri
top_users_ratings




#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

top_users_ratings.head() # movie'ler coklamıs durumda 6 benzer kullanıcı oldugu için.

# Adım 2: Film id’sive her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
# dataframe oluşturunuz.

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"}) # filmleri tekilleştirelim
recommendation_df.head()

recommendation_df = recommendation_df.reset_index()

# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.

recommendation_df[recommendation_df["weighted_rating"] > 3.5]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)[0:5]
movies_to_be_recommend.head()

# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.

movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"] # film isimleri gorebılmek adına
# movies_to_be_recommend zaten sıralı oldugu için en üstte gelen film ilk önerimiz olur.



#############################################
# Adım 6: Item-Based Recommendation
#############################################

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
user = 108170

# Adım 1: movie,rating veri setlerini okutunuz.

import pandas as pd

rating = pd.read_csv(r"datasets/rating.csv")

movie = pd.read_csv(r"datasets/movie.csv")

# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.

movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0] # film_id'sini tutmak istiyoruz.
movie_id

# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.

commen_movies_df[movie[movie["movieId"] == movie_id]["title"]]
movie_df = commen_movies_df[movie[movie["movieId"] == movie_id]["title"].values[0]]

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.

commen_movies_df.head() # tum kullanıcıların filmlere ait oyları ( var veya yok)
commen_movies_df.corrwith(movie_df).sort_values(ascending=False).head(10)



# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.
# Son iki adımı uygulayan fonksiyon
def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.
movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)

# 1'den 6'ya kadar. 0'da filmin kendisi var. Onu dışarda bıraktık.
movies_from_item_based[1:6].index # 0. index kendisi.





