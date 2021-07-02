import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ag_df_raw = pd.read_csv("./age_gender_info.csv",encoding="UTF-8")


raw_df = pd.read_csv("./train.csv")

# 원본 보존
df = raw_df.copy()

df.columns = ["코드", "총세대수", "임대구분", "지역", "공급유형", "전용면적", "면적별세대수", "공가수", "자격유형", "보증금", "임대료",\
             "지하철역수","버스정류장수", "주차면수", "등록차량수"]

df.drop(["자격유형"],inplace=True, axis=1)

# 지하철역, 버스정류장은 NaN일시 0으로 처리
df["지하철역수"].fillna(0., inplace=True)
df["버스정류장수"].fillna(0., inplace=True)


# 임대상가, 공공분양의 경우 보증금이 NaN
df["공급유형"][df["보증금"].isnull()].unique()

# 임대 상가는 아파트에 상가가 붙어있는 특별한 경우이고 특수한 방법으로 NaN을 제거하는것이 필요할것
df["전용면적"][df["공급유형"]=="임대상가"].unique()


# 일단 0으로 치환 후 실수형으로 타입 변경
df["보증금"].fillna(0, inplace=True)
df["임대료"].fillna(0, inplace=True)
df["보증금"].replace("-",0, inplace=True)
df["임대료"].replace("-",0, inplace=True)
df["보증금"]=df["보증금"].astype(np.float64)
df["임대료"]=df["임대료"].astype(np.float64)


# 공공분양의 경우 같은 지역 아파트들의 평균보증금, 임대료로 입력
locs_ = {}
locs2_ = {}
NaNList = {}
for loc in df["지역"].unique():
    locs_[loc] = df["보증금"][df["지역"] == loc][df["보증금"] != 0].mean()
    locs2_[loc] = df["임대료"][df["지역"] == loc][df["임대료"] != 0].mean()
    NaNList[loc]=df[df["보증금"]==0][df["공급유형"]=="공공분양"][df["지역"]==loc].index.tolist()

for loc in df["지역"].unique():
    for index in NaNList[loc]:
        df.보증금[index] = round(locs_[loc],0)
        df.임대료[index] = round(locs2_[loc],0)


df["전용면적"] = df["전용면적"].apply(lambda x:round(x/5,0)*5)
df["전용면적"]=df["전용면적"].astype(np.int32)

loc_dict = { key: num for num, key in enumerate(df['지역'].unique().tolist())}

supply_dict = { key: num for num, key in enumerate(df['공급유형'].unique().tolist())}


houses_dict = {key:[0 for i in range(10,90,5)] for key in df["코드"].unique()}

df.loc[:,["코드","공급유형","전용면적","면적별세대수"]]

housing_dict = {house:idx for idx,house in enumerate(range(10,90,5))}

code_dict = {key:idx for idx, key in enumerate(df['코드'].unique().tolist())}


ndf = pd.DataFrame(columns=["총세대수", "지역", "공급유형", "상가수",
                            "공가수", "보증금", "임대료", "지하철역수", "버스정류장수", "주차면수", "등록차량수"])
count=0
for code in df["코드"].unique():
    count += 1
    print(count)
    for i in df[["전용면적","면적별세대수"]][df["코드"]==code][df["공급유형"]!="임대상가"].iterrows():
        houses_dict[code][housing_dict[i[1][0]]] += i[1][1]

    temp_df = df[df["코드"]==code]

    temp ={
           "총세대수": temp_df["총세대수"].iloc[0],

           "지역": loc_dict[temp_df["지역"].iloc[0]], 

           "공급유형":supply_dict[temp_df["공급유형"].iloc[0]], 

           "상가수": temp_df[df["공급유형"]=="임대상가"].count()[0],

           "공가수": temp_df["공가수"].iloc[0], 
           
           "보증금": temp_df["면적별세대수"]*temp_df["보증금"].sum()//temp_df["총세대수"].iloc[0], 
           
          "공급유형":supply_dict[temp_df["공급유형"].iloc[0]], 

           "상가수": temp_df[df["공급유형"]=="임대상가"].count()[0],

           "공가수": temp_df["공가수"].iloc[0], 
           
           "보증금": (temp_df["면적별세대수"]*temp_df["보증금"]).sum()//temp_df["총세대수"].iloc[0], 
           
           "임대료": (temp_df["면적별세대수"]*temp_df["임대료"]).sum()//temp_df["총세대수"].iloc[0], 
           
           "지하철역수": temp_df["지하철역수"].iloc[0], 
           
           "버스정류장수": temp_df["버스정류장수"].iloc[0], 
           
           "주차면수": temp_df["주차면수"].iloc[0], 
           
           "등록차량수": temp_df["등록차량수"].iloc[0]
           }
    ndf = ndf.append(temp, ignore_index=True)

ndf

# train_test_split
import seaborn as sns

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(ndf.drop(["등록차량수"], axis=1), ndf["등록차량수"],test_size=.3, random_state=333)


# train
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(3)
tf.compat.v1.set_random_seed(3)

model = Sequential()
model.add(Dense(128, input_dim = 10, activation = 'relu'))
model.add(Dense(64, input_dim = 5, activation = 'relu'))
model.add(Dense(32, input_dim = 2, activation = 'relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=3000, batch_size = 16)


# 평가
from keras.metrics import MeanSquaredError

y_hat = model.predict(x_test)
x_test



print(met.result().numpy())


x_test
temp = y_test.copy()

temp = pd.DataFrame(temp)

temp["temp"] = y_hat
y_test
temp


temp["답"] = y_hat
plt.figure(figsize=(10, 5))
ax1 = sns.kdeplot(y_test, label="y_test")
ax2 = sns.kdeplot(y_hat, label="y_hat", ax=ax1)
plt.legend()
plt.show()
