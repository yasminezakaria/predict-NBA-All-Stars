import pandas as pd
import numpy as np

filename = "GuardsWesternConference"
# filename = "FrontcourtEasternConference"
# read csv file
historical = pd.read_csv("./2018-19 Seasons/"+filename+".csv")
all_star = pd.read_csv("./FinalData/allStar.csv")

names = []
for row in all_star['Starters']:
    names.append(row.split('\\')[0])
#
all_star['Starters'] = names
#
# print all_star.head(10)

status = []
for index1, row1 in historical.iterrows():
    s = 0
    for index2, row2 in all_star.iterrows():
        playerName = row1['Player']
        season = row1['Season'] + 1
        if str(row1['Player']).endswith('*'):
            playerName = playerName.replace("*", "")
            # print playerName
        if row2['Year'] == season and row2['Starters'] == playerName:
            s = 1
            break
    status.append(s)

historical['Status'] = status
print historical.query('Status == 1')

# write updated data frame to a new csv file
historical.to_csv("./2018-19 Seasons/"+filename+".csv",index = False)
