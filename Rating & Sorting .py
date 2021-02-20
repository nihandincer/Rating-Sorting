
########################################
# PROJE: Rating Product & Sorting Reviews in Amazon
########################################


import pandas as pd
import scipy.stats as st
import math
import os
os.getcwd()


df_=pd.read_csv("/Users/nHn/Desktop/df_sub.csv")
df=df_.copy()

from Desktop.helpers import check_df
check_df(df)



df["overall"].value_counts()

df["overall"].mean()  #4.58

# current day ve days değişkenleri oluşturma.
df['reviewTime'] = pd.to_datetime(df['reviewTime'], dayfirst=True)
current_day=pd.to_datetime("2021-02-12")
df["days"] = (current_day- df['reviewTime']).dt.days

#Zamanı çeyrek değerlere göre böldüm.
a=df["days"].quantile(0.25)
b=df["days"].quantile(0.50)
c=df["days"].quantile(0.75)

#zamana göre ağırlık verildi.
df.loc[df["days"] <= a, "overall"].mean() * 28 / 100 + \
    df.loc[(df["days"] > a) & (df["days"] <= b), "overall"].mean() * 26 / 100 + \
    df.loc[(df["days"] > b) & (df["days"] <= c), "overall"].mean() * 24 / 100 + \
    df.loc[(df["days"] > c), "overall"].mean() * 22 / 100  #4.59


#hrlpful değişkeni helpful_yes,helpful_no,helpful_vote olacak şekilde
# üc değişkene bölündü.
df["helpful"].value_counts()

new_features= df["helpful"].str.split(",", expand=True)


new_features = new_features.astype("string")
helpful_yes = new_features[0].str.lstrip("[")
helpful_yes = helpful_yes.astype("float64")

total_vote = new_features[1].str.rstrip("]")
total_vote = total_vote.astype("float64")

helpful_no= total_vote- helpful_yes


df["helpful_yes"]=helpful_yes
df["helpful_no"]=helpful_no



#Pozitif ve negatif yoruma göre sıralama.
def score_pos_neg_diff(pos, neg):
    return pos - neg


df["score_pos_neg_diff"] =df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"],x["helpful_no"]),axis=1)



#yorumların tutarlılığı için pozitif oranı bulma.
def score_average_rating(pos, neg):
    if pos + neg == 0:
        return 0
    return pos / (pos + neg)

df["score_average_rating"] =df.apply(lambda x: score_average_rating(x["helpful_yes"],x["helpful_no"]),axis=1)


#wilson lower  bound yöntemi.
def wilson_lower_bound(pos, neg, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not: Eğer skorlar 1-5 arasıdaysa 1-3 down, 4-5 up olarak işaretlenir ve bernoulli'ye uygun hale getirilir.

    Parameters
    ----------
    pos: int
        pozitif yorum sayısı
    neg: int
        negatif yorum sayısı
    confidence: float
        güven aralığı

    Returns
    -------
    wilson score: float

    """
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] =df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],x["helpful_no"]),axis=1)




#bayesian_rating
def bayesian_rating_products(n, confidence=0.95):
    """
    N yıldızlı puan sisteminde wilson lower bound score'u hesaplamak için kullanılan fonksiyon.
    Parameters
    ----------
    n: list or df
        puanların frekanslarını tutar.
        Örnek: [2, 40, 56, 12, 90] 2 tane 1 puan, 40 tane 2 puan, ... , 90 tane 5 puan.
    confidence: float
        güven aralığı

    Returns
    -------
    BRP score: float
        BRP ya da WLB skorları

    """
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


brp=df[["score_pos_neg_diff", "score_average_rating","wilson_lower_bound"]].sort_values("wilson_lower_bound", ascending=False).head(20)

brp_score= bayesian_rating_products(brp.index)
df["brp_score"]=brp_score



#total score
df["total_score"]= (df["score_average_rating"] * 25 / 100 +
                  df["wilson_lower_bound"] * 27 / 100 +
                  df["score_pos_neg_diff"] * 20 / 100+
                  df["brp_score"]* 28/100)

df["total_score"].sort_values(ascending=False).head(20)


