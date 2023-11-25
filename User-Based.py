#######################################################
#User-Based Collaborative Filtering
#######################################################

# Adım 1 : Veri setinin hazırlanması
# Adım 2 : Öneri yapılacak kullanıcının izlediği filmlerin belirlenmesi
# Adım 3 : Aynı filmleri izleyen kullanıcıların verisine ve id sine erişmek
# Adım 4 : Öneri yapılacak kullanıcı ile en benzer davranışlı kullanıcıların belirlenmesi
# Adım 5 : Weighted Average Recommendation Score Hesabı
# Adım 6 : Çalışmanın fonksiyonlaştırılması

####################################################
# Veri setining hazırlanması
####################################################

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv(r"datasets/movie.csv")
    rating = pd.read_csv(r"datasets/rating.csv")
    df = movie.merge(rating, how = "left", on = "movieId")
    rating_counts = pd.DataFrame(df["title"].value_counts()).reset_index()

    rating_counts.columns = ["title", "count"]

    # 1000 veya daha az inceleme alan nadir filmleri bulun
    rare_movies = rating_counts[rating_counts["count"] <= 1000]["title"]

    # Nadir filmleri hariç tutarak yeni bir veri çerçevesi oluşturun
    common_movies = df[~df["title"].isin(rare_movies)]

    # Her kullanıcının filmlerle ilgili derecelendirmelerini içeren bir çapraz tabloyu oluşturun
    user_movie_df = common_movies.pivot_table(index="userId", columns="title", values="rating")


    return user_movie_df
movie = pd.read_csv(r"datasets/movie.csv")
rating = pd.read_csv(r"datasets/rating.csv")
user_movie_df = create_user_movie_df()

import pandas as pd

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state= 45).values)

random_user_df = user_movie_df[user_movie_df.index == random_user]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# Adım 3: Aynı filmleri izleyen kullanıcıların verisine ve ID'sine erişme
movies_watched_df = user_movie_df[movies_watched]


# Adım 4: Öneri yapılacak kullanıcı ile en benzer davranışlı kullanıcıların belirlenmesi
final_df = pd.concat([movies_watched_df, random_user_df])

corr_df = final_df.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ["user_id_1", "user_id_2"]

corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] > 0.65)][["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by="corr", ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

# Adım 5: Weighted Average Recommendation Score Hesabı
rating = pd.read_csv(r"datasets/rating.csv")

top_users_rating = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")

top_users_rating = top_users_rating[top_users_rating["userId"] != random_user]

top_users_rating["weighted_rating"] = top_users_rating["corr"] * top_users_rating["rating"]

recommendation_df = top_users_rating.groupby("movieId").agg({"weighted_rating": "mean"}).reset_index()

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5 ]

movie = pd.read_csv(r"datasets/movie.csv")
movies_to_be_recommend.merge(movie[["movieId","title"]])