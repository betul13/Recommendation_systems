####################################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
####################################################
# 1- Veri Ön İşleme
# 2- APL Veri Yapısını Hazırlama (Invoice-Product Matrix)
# 3- Birliktelik Kurallarının çıkarılması
# 4- Çalışmanın Scriptini Hazırlama
# 5- Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

#######################################################
# VERİ ÖN İŞLEME
#######################################################

import pandas as pd
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.width",500)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel(r"datasets/online_retail_II.xlsx", sheet_name= "Year 2010-2011")

df = df_.copy()

#pip install openpyxl engine = "openpyxl"

df.head()

df.describe().T
df.isnull().sum()
df.shape



def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit,up_limit

def replace_with_tresholds(dataframe,variable):
    low_limit, up_limit = outlier_thresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable] < low_limit),variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit),variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace = True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C",na = False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_tresholds(dataframe,"Quantity")
    replace_with_tresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)

df.head()

#########################################################
# ARL Veri Yapılarını Hazırlamak (Preparing ARL Data Structures (invoice-product matrix)
#########################################################

#ürün var yok 0-1 olsun istiyoruz

df_fr = df[df["Country"] == "France"]

df_fr.groupby(["Invoice","Description"]).agg({"Quantity":"sum"}).unstack().fillna(0).applymap(lambda x : 1 if x > 0 else 0)
#unstack ile sütunlar değişken haline geldi
# #applymap bütün gözlemleri gezer.

def create_invoice_product_df(dataframe, id = False):
    if id :
        return dataframe.groupby(["Invoice", "StockCode"]).agg({"Quantity": "sum"}).unstack().fillna(0).\
         applymap(lambda x: 1 if x > 0 else 0)  # applymap bütün gözlemleri gezer

    else :
        return dataframe.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().fillna(0).\
         applymap(lambda x: 1 if x > 0 else 0)

fr_inv_pro_df = create_invoice_product_df(df_fr,id = True)

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_fr,10120)


#######################################################
#Birliktelik Kurallarının Çıkarılması
#######################################################

frequent_itemsets = apriori(fr_inv_pro_df,
                            min_support = 0.01,
                            use_colnames = True )#olası ürün birlikteliklerinin supportu yani olasılığı

frequent_itemsets.sort_values("support",ascending = False )

rules = association_rules(frequent_itemsets,
                  metric = "support",
                  min_threshold = 0.01)

rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.1) & (rules["lift"] > 5)].sort_values("confidence",ascending=False)

product_id = 22492
sorted_rules = rules.sort_values
recommendation_list = []

for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id :
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

print(recommendation_list[0])