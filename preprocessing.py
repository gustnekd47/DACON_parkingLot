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


# ndf 편집

ndf = pd.DataFrame(columns=["코드", "총세대수", "지역", "공급유형", "상가수",
                            "공가수", "보증금", "임대료", "지하철역수", "버스정류장수", "주차면수", "등록차량수"])


for code in df["코드"].unique():
    for i in df[["전용면적","면적별세대수"]][df["코드"]==code].iterrows():
        houses_dict[code][housing_dict[i[1][0]]] += i[1][1]

    print(i[1][0], i[1][1]) # 전용면적, 면적별세대수
    
    temp ={
           "코드": code, 

           "총세대수": df["총세대수"][df["코드"]==code][0], 

           "지역":loc_dict[df["지역"][df["코드"]==code][0]], 

           "공급유형":supply_dict[df["지역"][df["코드"]==code][0]], 

           "상가수": df[df["코드"]==code][df["공급유형"]=="임대상가"].count()[0],

           "공가수": df["공가수"][df["코드"]==code][0], 
           
           "보증금":1, 
           
           "임대료": 1, 
           
           "지하철역수": 1, 
           
           "버스정류장수": 1, 
           
           "주차면수": 1, 
           
           "등록차량수": 1
           }