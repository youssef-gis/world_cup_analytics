import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os 
import pathlib, requests
import warnings
from scipy import stats
from mplsoccer import PyPizza, FontManager

pd.options.display.max_columns = None
pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore")

#Opening wyscout data
train_df = pd.DataFrame()
# for i in range(0,13):
#     URL="https://raw.githubusercontent.com/soccermatics/Soccermatics/refs/heads/main/course/lessons/data/Wyscout/events_England_"+str(i+1)+".json"
#     response = requests.get(URL)
#     response.raise_for_status()
#     json_data = response.json()
#     train_df = pd.concat([train_df, pd.DataFrame(json_data)] , ignore_index = True)

# train_df = train_df.loc[train_df.apply(lambda x: len(x.positions) == 2, axis=1)]
# train_df.to_csv(f"Outputs/plot_xGModelFit.csv", index=False)
train = pd.read_csv("Outputs/plot_xGModelFit.csv", delimiter=',')
train.positions = train.positions.apply(lambda x: eval(x))
train.tags = train.tags.apply(lambda x: eval(x))

#print(train.matchId.unique())   
#print(train.subEventName.unique())
def calcXG(df, npXG):
        
    #calculate xg value with npXG for penalty shots 
    shots  = df.loc[df.eventName == 'Shot'].copy()
    shots['X'] = shots.positions.apply(lambda x: (100-x[0]['x'])*105/100 )
    shots['Y'] = shots.positions.apply(lambda x: (x[0]['y'])*68/100 )
    shots['C'] = shots.positions.apply(lambda x: abs(x[0]['y'] - 50)*68/100)

    shots['Distance'] = np.sqrt(np.square(shots['X']) + np.square(shots['C']))
    shots["Angle"] = np.where(np.arctan(7.32*shots["X"]/(shots["X"]**2+shots["C"]**2-(7.32/2)**2)) > 0, 
                    np.arctan(7.32*shots["X"]/(shots["X"]**2 + shots["C"]**2-(7.32/2)**2)),
                    np.arctan(7.32*shots["X"] /(shots["X"]**2+shots["C"]**2-(7.32/2)**2)) + np.pi)

    shots['Goal'] = shots.tags.apply(lambda x: 1 if {'id': 101} in x else 0).astype(object)
    #print(shots.Goal.value_counts())
    headers = shots.loc[shots.apply(lambda x:{'id': 403} in x.tags, axis=1)]
    nonheaders = shots.drop(headers.index)

    headers_model = smf.glm(formula="Goal ~ Distance + Angle", data=headers, 
                            family=sm.families.Binomial()).fit()

    nonheaders_model = smf.glm(formula="Goal ~ Distance + Angle", data=nonheaders, 
                            family=sm.families.Binomial()).fit()

    #assigning xG value to shots
    b_heads = headers_model.params
    xG = 1/(1+np.exp(b_heads[0]+b_heads[1]*headers['Distance']+b_heads[2]*headers['Angle']))
    headers= headers.assign(xG = xG)

    b_nonheads = nonheaders_model.params

    xG = 1/(1+np.exp(b_nonheads[0]+ b_nonheads[1]*nonheaders['Distance']+b_nonheads[2]*nonheaders['Angle']))
    nonheaders = nonheaders.assign(xG = xG)

    #adding xG of penaltie tp 0.8
    if npXG == False:
        penalties = df.loc[df.subEventName == 'Penalty'].copy()
        penalties = penalties.assign(xG = 0.8)
        all_shots_xG = pd.concat([nonheaders[['playerId', 'xG']], headers[['playerId', 'xG']], 
                                penalties[['playerId', 'xG']] ])
        xG_sum = all_shots_xG.groupby(['playerId'])['xG'].sum().sort_values(ascending=False).reset_index()
    else:
        all_shots_xG = pd.concat([nonheaders[['playerId', 'xG']], headers[['playerId', 'xG']]])
        all_shots_xG.rename(columns = {'xG': 'npXG'}, inplace = True)
        xG_sum = all_shots_xG.groupby(['playerId'])['npXG'].sum().sort_values(ascending=False).reset_index()
    
    return xG_sum

xG_sum = calcXG(train, npXG = True)
#print(xG_sum.head())


def FinalThird(df):
    df = df.copy()
    df['nextPlayerId']= df['playerId'].shift(-1)#.astype(int)
   
    passes = df.loc[train.eventName == 'Pass'].copy()
    #change coordinates
    passes['X'] = passes.positions.apply(lambda x: (x[0]['x']) * 105/100)
    passes['Y'] = passes.positions.apply(lambda y: (100 - y[0]['y']) * 68/100)
    passes['end_x'] = passes.positions.apply(lambda x: (x[1]['x']) * 105/100)
    passes['end_y'] = passes.positions.apply(lambda y: (100 - y[1]['y']) * 68/100)

    #accurates passes 
    accurate_passes = passes.loc[passes.apply(lambda x: {'id':1801} in x.tags, axis=1)]

    #passes into finale third 
    final_third_passes= accurate_passes.loc[accurate_passes['end_x'] > 2*105/3]

    #passes into final third by player
    ftp_player = final_third_passes.groupby(['playerId']).end_x.count().reset_index()
    ftp_player.rename(columns = {'end_x': 'final_third_passes'}, inplace = True)
    #print(ftp_player.head())

    #passes received in final third
    received_player = final_third_passes.groupby(['nextPlayerId']).end_x.count().reset_index()
    received_player.rename(columns = {'end_x': 'final_third_receptions',
                                     'nextPlayerId': 'playerId'}
                           , inplace = True)
    #print(received_player.head())
    #merge the two dataframes
    final_third = ftp_player.merge(received_player, how='outer', on=['playerId'])
    return final_third

final_third = FinalThird(train)
final_third = final_third.fillna(0)
#print(final_third.head())

def duel_wons(df):
    #ground and aerial duels won
    df = train.copy()
    air_duels = df.loc[df.subEventName == 'Air duel']
    #won air duels
    wons_air_duels = air_duels.loc[air_duels.apply(lambda x: {'id': 703} in x.tags, axis=1)]

    #group and sum won air duels by player
    won_air_duels_by_player = wons_air_duels.groupby(['playerId']).eventId.count().reset_index()
    won_air_duels_by_player.rename(columns = {'eventId': 'won_air_duels'}, inplace = True)

    #Find ground duels
    ground_duels = df.loc[df.subEventName == 'Ground attacking duel']
    ground_duels_won = ground_duels.loc[ground_duels.apply(lambda x: {'id': 703} in x.tags, axis=1)]
    ground_duels_won_by_player = ground_duels_won.groupby(['playerId']).eventId.count().reset_index()
    ground_duels_won_by_player.rename(columns = {'eventId': 'won_ground_duels'}, inplace = True)

    #outer joins
    duel_wons = ground_duels_won_by_player.merge(won_air_duels_by_player, how='outer', on=['playerId'])
    return duel_wons

duel_wons = duel_wons(train)
#print(duel_wons.head())

#smart passes
def smart_passes(df):

    smart_passes = df.loc[df['subEventName'] == 'Smart pass']
    succ_smart_passes = smart_passes.loc[smart_passes.apply(lambda x:{'id': 1801} in x.tags, axis=1)]

    #group by player and count smart passes
    smrt_passes_by_player = succ_smart_passes.groupby(['playerId']).eventId.count().reset_index()
    smrt_passes_by_player.rename(columns = {'eventId': 'smart_passes'}, inplace = True)

    return smrt_passes_by_player

smart_passes = smart_passes(train)
#print(smart_passes.head())

def Goal_Assist_Key_Passes(df):
    shots = df.loc[df.subEventName == 'Shot']
    goals = shots.loc[shots.apply(lambda x: {'id': 101} in x.tags, axis=1)]  
    passes = df.loc[df.eventName == 'Pass']
    assists = passes.loc[passes.apply(lambda x: {'id': 301} in x.tags, axis=1)]
    key_passes = passes.loc[passes.apply(lambda x: {'id': 302} in x.tags, axis=1)]

    #group by player and count goals; assists and key passes
    goals_by_player = goals.groupby(['playerId']).eventId.count().reset_index()
    goals_by_player.rename(columns = {'eventId': 'goals'}, inplace = True)
    
    assists_by_player = assists.groupby(['playerId']).eventId.count().reset_index()
    assists_by_player.rename(columns = {'eventId': 'assists'}, inplace = True)

    key_passes_by_player = key_passes.groupby(['playerId']).eventId.count().reset_index()
    key_passes_by_player.rename(columns = {'eventId': 'key_passes'}, inplace = True)

    goal_key_assist = goals_by_player.merge(assists_by_player, how='outer', on=['playerId']
                    ).merge(key_passes_by_player, how='outer', on=['playerId'])
    return goal_key_assist

data = Goal_Assist_Key_Passes(train).fillna(0)
#print(data.head()) 

#minutes played throughout the season by each player
resp = requests.get('https://raw.githubusercontent.com/soccermatics/Soccermatics/refs/heads/main/course/lessons/minutes_played/minutes_played_per_game_England.json')
resp.raise_for_status()
minutes_played = resp.json()
minutes_played = pd.DataFrame(minutes_played)
minutes = minutes_played.groupby(['playerId']).minutesPlayed.sum().reset_index()


#Calculating possession
possesion_dict = {}
for i, row in minutes_played.iterrows():
    player_id, team_id, match_id= row["playerId"], row["teamId"], row["matchId"]
    if not str(player_id) in possesion_dict.keys():
        possesion_dict[str(player_id)] = {'team_passes':0, 'all_passes':0}
    min_in_match = row["player_in_min"]*60
    min_out_match = row["player_out_min"]*60

    #get the events for a specific game
    match_event = train.loc[train['matchId'] == match_id].copy()
    #print("match_id : ", str(match_id))
    match_event.loc[match_event['matchPeriod'] == '2H', 'eventSec']= match_event.loc[match_event['matchPeriod']=='2H', 'eventSec'] + match_event.loc[match_event['matchPeriod'] == "1H"]['eventSec'].iloc[-1]

    #take the events for the player
    player_in_match = match_event.loc[match_event['eventSec']>min_in_match
                                      ].loc[match_event['eventSec']<=min_out_match]
    #take all passes and duels
    all_passes = match_event.loc[match_event['eventName'].isin(['Pass', 'Duel'])]
    if len(all_passes) > 0:
        no_contact = all_passes.loc[all_passes['subEventName'].isin(
            ["Air duel", "Ground defending duel","Ground loose ball duel"])].loc[
                all_passes.apply(lambda x: {'id':701} in x.tags, axis=1)]
        all_passes = all_passes.drop(no_contact.index)
    
    #take team passes
    team_passes = all_passes.loc[all_passes['teamId'] == team_id]
    possesion_dict[str(player_id)]['team_passes'] += len(team_passes)
    possesion_dict[str(player_id)]['all_passes'] += len(all_passes)

#calculate possession for each player
percentage_possesion = {key: value['team_passes']/value['all_passes'] 
                if value['all_passes'] > 0 else 0 for key, value in possesion_dict.items()}
#craete a dataframe
percent_df = pd.DataFrame(percentage_possesion.items(), columns = ['playerId', 'possession'])
percent_df['playerId'] = percent_df['playerId'].astype(int)

#merge all dataframes
players_ids = train.playerId.unique()
summary = pd.DataFrame(players_ids, columns = ['playerId'])
summary = summary.merge(xG_sum, how='left', on= ['playerId']).merge(
                        final_third, how='left', on=['playerId']).merge(
                        duel_wons, how='left', on=['playerId']).merge(
                        smart_passes, how='left', on=['playerId']).merge(
                        data, how='left', on=['playerId']).merge(
                        minutes, how='left', on=['playerId'])
summary = summary.fillna(0)
summary = summary.loc[summary.minutesPlayed > 400]

# print(summary.head())

#filtering positions
resp = requests.get('https://raw.githubusercontent.com/soccermatics/Soccermatics/refs/heads/main/course/lessons/data/Wyscout/players.json')
resp.raise_for_status()
players = resp.json()
players = pd.DataFrame(players)
forwards = players.loc[players.apply(lambda x: x.role['name']=='Forward', axis=1)]
forwards.rename(columns = {'wyId': 'playerId'}, inplace = True)
forwards = forwards[['playerId', 'shortName']]

summary = summary.merge(forwards, how='inner', on=['playerId'])
summary = summary.merge(percent_df, how='left', on=['playerId'])
#print(summary.head())
#stats per 90 minutes 
summary_90 = pd.DataFrame()
summary_90['shortName'] = summary['shortName']

#summary adjusted for ball possession
summary_adjusted= pd.DataFrame()
summary_adjusted['shortName'] = summary['shortName']
for column in summary.columns[1:-2]:
    summary_90[column+ '_per_90min']=summary.apply(lambda x:x[column]*90/x['minutesPlayed'],axis=1)

for column in summary.columns[1:-2]:    
    summary_adjusted[column+ '_per_90min'] = summary.apply(lambda x:(x[column]/x['possession'])*90/x['minutesPlayed']*x['possession'],axis=1)
#print(summary_90.head())


#stat values for a player
Salah_stats = summary_90.loc[summary_90.shortName == 'Mohamed Salah']
Salah_stats_adjusted = summary_adjusted.loc[summary_adjusted.shortName == 'Mohamed Salah']

#take specific columns
per_90min_columns = Salah_stats.columns[1:]
adjusted_columns = Salah_stats_adjusted.columns[1:-1]
print( adjusted_columns)
values = [round(Salah_stats[column].iloc[0], 2) for column in per_90min_columns]
adjusted_values = [Salah_stats_adjusted[column].iloc[0] for column in adjusted_columns]

# for column_name, value in zip(per_90min_columns, values):
#     print(column_name, value)
#percentiles
percentiles = [int(stats.percentileofscore(summary_90[column], Salah_stats[column].iloc[0])) for column in per_90min_columns]
adjusted_percentiles = [int(stats.percentileofscore(summary_adjusted[column],
                 Salah_stats_adjusted[column].iloc[0])) for column in adjusted_columns]


#radar plot
#list of names in the radar plot
names = ["non-penalty Expected Goals","Passes Ending in Final Third",
        "Passes Received in Final Third","Offensive Ground Duels Won",
        "Air Duels Won", "Smart Passes","non-penalty Goals","Assists", "Key Passes" ]

#print(len(names), len(percentiles), len(adjusted_percentiles))
# for name, percent in zip(names, adjusted_percentiles):
#     print(name, percent)

# print(names)
# print(adjusted_columns)
# print(adjusted_percentiles)

slice_colors = ["blue"] * 2 + ["green"] * 5 + ["red"] * 2
text_colors = ["white"]*9
#plotting the radar plot
baker = PyPizza(
    params=names,
    min_range=None,
    max_range=None,
    straight_line_color="#000000",
    straight_line_lw=1,
    last_circle_lw=1,
    other_circle_lw=1,
    other_circle_ls="-.",
)

fig, axs = baker.make_pizza(
    adjusted_percentiles,
    #percentiles,
    figsize=(10, 10),
    param_location=105,
    slice_colors=slice_colors,
    value_colors=text_colors,
    value_bck_colors=slice_colors,
    kwargs_slices=dict(
        facecolor="cornflowerblue", edgecolor="#000000",
        zorder=2, linewidth=1
    ),                   # values to be used when plotting slices
    kwargs_params=dict(
        color="#000000", fontsize=12, va="center"
    ),                   # values to be used when adding parameter
    kwargs_values=dict(
        color="#000000", fontsize=12,
        bbox=dict(
            edgecolor="#000000", facecolor="cornflowerblue",
            boxstyle="round,pad=0.2", lw=1
        )
    )                    # values to be used when adding parameter-values
)

#putting text
texts = baker.get_value_texts()
# for i, text in enumerate(texts):
#     text.set_text(str(values[i]))
# add title
fig.text(
    0.315, 0.96, "Mohammed Salah per 90 min - Liverpool FC", 
    size=18,ha="center", color="#000000"
)

# add subtitle
fig.text(
    0.515, 0.93,
     "Percentile Rank vs Premier League Forwards | Season 2017-18",
    size=15,
    ha="center", color="#000000"
)

fig.set_size_inches(10, 10)
fig.savefig(f'Outputs/Mohammed_Salah_per_90_min_possesion_adjusted_Liverpool_FC.pdf', dpi=100)

plt.show()