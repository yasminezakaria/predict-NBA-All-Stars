import pandas as pd

GE = "GuardsEasternConference.csv"
GW = "GuardsWesternConference.csv"

FW = "FrontcourtWesternConference.csv"
FE = "FrontcourtEasternConference.csv"
# data_1 = pd.read_csv('./ReadyFilescsv/'+f1)
# data_2 = pd.read_csv('./ReadyFilescsv/'+f2)
# combined_files = pd.concat([data_1,data_2],ignore_index=True)
# combined_files.to_csv("./combined_files/frontcourt.csv", index=False, encoding='utf-8-sig')

data_GE = pd.read_csv('./ReadyFilescsv/GuardsEasternConference.csv')
data_GW = pd.read_csv('./ReadyFilescsv/GuardsWesternConference.csv')
data_FE = pd.read_csv('./ReadyFilescsv/FrontcourtEasternConference.csv')
data_FW = pd.read_csv('./ReadyFilescsv/FrontcourtWesternConference.csv')


EF_coulms = ['MP_tot','MP','3P_perc','TRB_tot','3PAr','ORB_per_G','BLK_tot','DRB_per_100p','STL_per_100p','STL_perc','BLK_per_100p',
'PF_per_100p','DBPM','3PA_per_36m','3PA_per_100p','DRB_perc','Age','ORB_tot','DRtg','ORB_per_100p','BLK_per_36m','PF_per_G','G',
             'TS_perc','TRB_per_100p','FTr','ORtg','FT_perc','PF_tot','TOV_perc','DRB_per_36m','ORB_per_36m','BLK_perc','ORB_perc',
             'TRB_perc','eFG_perc','2P_perc','FG_perc','TRB_per_36m']

WF_columns = ['DRB_per_36m','3PA_per_G','3P_perc','STL_tot','3PA_per_36m','FTr',
'PF_per_36m','3PA_per_100p','DRB_perc','3PAr','BLK_per_36m','TOV_perc','DBPM','PF_per_100p','TRB_per_100p','ORtg',
'FG_perc','ORB_tot','3PA_tot','BLK_perc','ORB_per_36m','BLK_per_100p','PF_per_G','G','PF_tot','eFG_perc','2P_perc',
'TRB_per_36m','TRB_perc','TS_perc','ORB_perc', 'ORB_per_100p','FT_perc','Age','STL_perc','STL_per_100p','STL_per_36m']

EG_columns = ['DRB_tot','OWS','AST_perc','MP_tot','MP','ORB_per_100p','BLK_perc','Tm_Rank','STL_per_36m',
'FTr','PF_per_100p','ORB_per_36m','PF_per_36m','STL_perc','STL_per_100p','ORtg','PF_per_G','Age','3PA_per_G','3P_per_G',
              'AST_per_100p','FG_perc','3P_perc','AST_per_36m',
'FT_perc','TRB_perc','3P_per_36m','ORB_perc','DRtg','3PA_tot','PF_tot','TRB_per_36m','3PA_per_100p','TOV_perc','3P_tot','TS_perc',
'3PAr','G','DRB_perc','3PA_per_36m','TRB_per_100p','2P_perc','DRB_per_100p','DBPM','3P_per_100p','eFG_perc','DRB_per_36m']

WG_columns = ['ORtg','TRB_per_100p','TOV_per_100p','AST_per_36m','FTr','AST_per_100p','FT_perc','DBPM','2PA_per_100p','PF_per_36m','ORB_per_100p',
'BLK_per_100p','STL_per_36m','PF_tot','PF_per_G','PF_per_100p','TOV_per_36m','2P_perc','TS_perc','FG_perc','Age','3P_per_100p','STL_perc',
'BLK_perc','3PA_per_36m','3P_per_36m','3PAr','STL_per_100p','3PA_per_100p','eFG_perc','ORB_perc','G','3P_perc','TOV_perc','DRtg']

edited_data_GE = data_GE.drop(EG_columns, axis=1)

edited_data_GW = data_GW.drop(WG_columns, axis=1)

edited_data_FE = data_FE.drop(EF_coulms, axis=1)

edited_data_FW = data_FW.drop(WF_columns, axis=1)


print "FE: " + str(len(edited_data_FE.columns))
print "FW: " + str(len(edited_data_FW.columns))
print "GE: " + str(len(edited_data_GE.columns))
print "GW: " + str(len(edited_data_GW.columns))


edited_data_FE.to_csv('./Edited_files/FrontcourtEasternConference.csv', index=False)
edited_data_GE.to_csv('./Edited_files/GuardsEasternConference.csv', index=False)
edited_data_FW.to_csv('./Edited_files/FrontcourtWesternConference.csv', index=False)
edited_data_GW.to_csv('./Edited_files/GuardsWesternConference.csv', index=False)
