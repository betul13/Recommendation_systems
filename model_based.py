##############################################################
# Model-Based Collaborative Filtering : Matrix Factorization
##############################################################

import pandas as pd
from surprise import Reader,SVD,Dataset,accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option("display.max_columns", None)

# Adım 1- Veri setinin hazırlanması
# Adım 2- Modelleme
# Adım 3- Model Tuning
# Adım 4- Final Model ve Tahmin

#Veri Setinin Hazırlanması

movie = pd.read_csv(r"/content/movie.csv")
rating = pd.read_csv(r"/content/rating.csv")

df = movie.merge(rating, how = "left", on = "movieId")

df.head()
movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]
sample_df = df[df.movieId.isin(movie_ids)]
sample_df.head()
user_movie_df = sample_df.pivot_table(index = ["userId"],
                                      columns = ["title"],
                                      values = ["rating"] )


reader = Reader(rating_scale = (1,5))  #skala vermemiz lazım. Hesabı neye göre yapacağını belirlemek için

data  = Dataset.load_from_df(sample_df[["userId","movieId","rating"]],reader) #hesap yapacağı şekle dönüştürürüz.

#######################
# MODELLEME
#######################

train_set, test_set = train_test_split(data, test_size = 0.25)

svd_model = SVD() #model nesnesi yöntemi kullanacağımız fonksiyon

svd_model.fit(train_set) #train set üzerinden öğren

predictions = svd_model.test(test_set)

accuracy.rmse(predictions)

svd_model.predict(uid = 1.0, iid = 356, verbose = True)

#######################################################################################
# MODEL TUNING (modelin tahmin performansını artırmak,hiperparametreleri optimize etmek)
########################################################################################

param_grid = {"n_epochs" : [5, 10, 20],
              "lr_all" : [0.002, 0.005, 0.007]}

gs = GridSearchCV(SVD,
                  param_grid,
                  measures = ["rmse","mae"],
                  cv = 3, n_jobs = -1, joblib_verbose =True )

gs.fit(data)

gs.best_score["rmse"]
gs.best_params["rmse"]

###################################################
# FINAL MODEL ve TAHMIN
###################################################

dir(svd_model)

svd_model.n_epochs

svd_model = SVD(**gs.best_params["rmse"])

reader = Reader(rating_scale = (1,5))  #skala vermemiz lazım. Hesabı neye göre yapacağını belirlemek için

data  = Dataset.load_from_df(sample_df[["userId","movieId","rating"]],reader) #hesap yapacağı şekle dönüştürürüz.

data = data.build_full_trainset()

svd_model.fit(data)

svd_model.predict(uid = 1.0, iid = 541, verbose = True)