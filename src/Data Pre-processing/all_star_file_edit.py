import pandas as pd
import numpy as np

# filename = "GuardsEasternConference"
filename = "player_data_1999-2019"
# read csv file
historical = pd.read_csv("./FinalData/" + filename + ".csv")

# edit Season col to int64 instead of range
season = []

for row in historical['Season']:
    season.append(row[:4])

historical['Season'] = season
historical['Season'] = historical['Season'].astype('int64')

historical = historical.query("Season <= 2019 and Season >= 2017")

western_conf = historical.query("Tm == 'GSW' or Tm == 'UTA' or Tm == 'DEN' or Tm == 'HOU' or Tm == 'POR' or Tm == 'OKC'"
                                "or Tm == 'SEA' or Tm == 'SAS' or Tm == 'LAC' or Tm == 'NOP' or Tm == 'NOK' or Tm == 'NOH'"
                                "or Tm == 'MIN' or Tm == 'SAC' or Tm == 'LAL' or Tm == 'DAL'"
                                "or Tm == 'MEM' or Tm == 'VAN' or Tm == 'PHO'")

eastern_conf = historical.query("Tm == 'MIL' or Tm == 'TOR' or Tm == 'BOS' or Tm == 'PHI' or Tm == 'IND' or Tm == 'MIA'"
                                "or Tm == 'DET' or Tm == 'ORL' or Tm == 'BRK' or Tm == 'NJN' or Tm == 'CHH' or Tm == 'CHO'"
                                "or Tm == 'CHA' or Tm == 'WAS' or Tm == 'ATL' or Tm == 'CHI'"
                                "or Tm == 'NYK' or Tm == 'CLE'")

eastern_guards = eastern_conf.query("Pos == 'PG' or Pos == 'SG'")
western_guards = western_conf.query("Pos == 'PG' or Pos == 'SG'")

eastern_frontcourt = eastern_conf.query("Pos == 'C' or Pos == 'PF' or Pos == 'SF' or Pos == 'F'")
western_frontcourt = western_conf.query("Pos == 'C' or Pos == 'PF' or Pos == 'SF' or Pos == 'F'")

eastern_guards.to_csv("./2018-19 Seasons/GuardsEasternConference" + ".csv", index=False)
western_guards.to_csv("./2018-19 Seasons/GuardsWesternConference" + ".csv", index=False)

eastern_frontcourt.to_csv("./2018-19 Seasons/FrontcourtEasternConference" + ".csv", index=False)
western_frontcourt.to_csv("./2018-19 Seasons/FrontcourtWesternConference" + ".csv", index=False)

# all_star = pd.read_csv("./UpdatedProjectDatacsv/All-starWestern.csv")
