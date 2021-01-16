import pandas as pd
import numpy as np

# filename = "GuardsWesternConference"
filename = "FrontcourtWesternConference"
# read csv file
df = pd.read_csv("./2018-19 Seasons/"+filename+".csv")

# replace team names with their rankings
dict
teamsRankWesternConf = {'GSW':1,
                       'UTA':2,
                       'DEN':3,
                       'HOU':4,
                       'POR':5,
                       'OKC':6, 'SEA':6,
                       'SAS':7,
                       'LAC':8,
                       'NOP':9, 'NOK':9, 'NOH':9,
                       'MIN':10,
                       'SAC':11,
                       'LAL':12,
                       'DAL':13,
                       'MEM':14, 'VAN':14,
                       'PHO':15}
teamsRankEasternConf = {'MIL':1,
                       'TOR':2,
                       'BOS':3,
                       'PHI':4,
                       'IND':5,
                       'MIA':6,
                       'DET':7,
                       'ORL':8,
                       'BRK':9, 'NJN':9,
                       'CHH':10, 'CHO':10, 'CHA':10,
                       'WAS':11,
                       'ATL':12,
                       'CHI':13,
                       'NYK':14,
                       'CLE':15}
teamRank = []
for index, row in df.iterrows():
    teamRank.append(teamsRankWesternConf[row['Tm']])

df['Tm_Rank'] = teamRank
print df.head(10)
# df["Tm"] = df["Tm"].astype('category')
# df["Tm_label"] = df["Tm"].cat.codes

# label encode Pos col
# df["Pos"] = df["Pos"].astype('category')
# df["Pos_label"] = df["Pos"].cat.codes

# drop Player, Pos and Tm columns
df = df.drop(['Tm', 'Pos', "Player"], axis=1)

# write updated data frame to a new csv file
df.to_csv('./2018-19 Seasons/'+filename+'.csv', index=False)
