import pandas as pd

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.width",500)
from mlxtend.frequent_patterns import apriori, association_rules


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

def create_invoice_product_df(dataframe, id = False):
    if id :
        return dataframe.groupby(["Invoice", "StockCode"]).agg({"Quantity": "sum"}).unstack().fillna(0).\
         applymap(lambda x: 1 if x > 0 else 0)  # applymap bütün gözlemleri gezer

    else :
        return dataframe.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().fillna(0).\
         applymap(lambda x: 1 if x > 0 else 0)

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


def create_rules(dataframe,id = True,country = "France"):
    dataframe = dataframe[dataframe["Country"] == country]
    dataframe = create_invoice_product_df(dataframe,id)
    frequent_itemsets = apriori(dataframe,
                                min_support=0.01,
                                use_colnames=True)  # olası ürün birlikteliklerinin supportu yani olasılığı

    rules = association_rules(dataframe,
                              metric="support",
                              min_threshold=0.01)
    return rules


df_ = pd.read_excel(r"datasets/online_retail_II.xlsx", sheet_name= "Year 2010-2011")

df = df_.copy()

df = retail_data_prep(df)
df_final = create_rules(df)
df_final.head()

##################################################################################
#Sepet aşamasındaki kullanıcılara ürün önerisinde bulunmak
##################################################################################

#örnek : kullanıcı örnek ürün id : 22492

product_id = 22492
sorted_rules = df_final.sort_values("lift",ascending = False)
recommendation_list = []

for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id :
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))


print(recommendation_list[0:3])