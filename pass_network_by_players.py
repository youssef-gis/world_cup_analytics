from matplotlib.colors import to_rgba
import pandas as pd
from mplsoccer import Pitch, Sbopen, FontManager, VerticalPitch
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

pd.set_option("display.max_columns", None)
rcParams['text.color'] = '#c7d5cc'  # set the default text color

# fontmanager for Google font (robotto)
robotto_regular = FontManager()
parser =Sbopen()
comp_df=parser.competition()

#Find the world cup competition games
wc_mask = comp_df[ (comp_df['competition_name']=='FIFA World Cup') & (comp_df['season_id'] == 106)]
matches = parser.match(competition_id=43, season_id=106)

#filter by finding morocco games
morocco_mask = (matches["home_team_name"]=='Morocco')  | (matches["away_team_name"]=='Morocco')
morocco_matchs = matches[morocco_mask]
#Find Morocco games ids
morocco_games_id = [game.match_id for i, game in morocco_matchs.iterrows()]
#print(morocco_games_id)

#get events of the first game
events_df, realted, freeze, players = parser.event(match_id=morocco_games_id[2])

#filter events by team name
team_names= events_df.team_name.unique()

#check for index of first sub
#first_sub_index = events_df[(events_df['type_name']=='Substitution') & (events_df['team_name']=="Morocco")].iloc[0]['index']

morocco_passes_df = events_df.loc[(events_df['team_name']=="Morocco") &
                               # (events_df['index'] < first_sub_index ) & 
                                (events_df['type_name']=='Pass') & 
                                (events_df.sub_type_name != 'Throw-in'),['x', 'y', 'end_x',
                                'end_y', 'player_name', 'pass_recipient_name']]

scatter_df = pd.DataFrame()

for i, name in enumerate(morocco_passes_df['player_name'].unique()):
    pass_x = morocco_passes_df[morocco_passes_df['player_name']==name]['x'].to_numpy()
    recp_x = morocco_passes_df[morocco_passes_df['pass_recipient_name']==name]['end_x'].to_numpy()
    pass_y = morocco_passes_df[morocco_passes_df['player_name']==name]['y'].to_numpy()
    recp_y = morocco_passes_df[morocco_passes_df['pass_recipient_name']==name]['end_y'].to_numpy()

    scatter_df.at[i, "player_name"] = name#.split(' ')[-1]
    scatter_df.at[i, 'x'] = np.mean(np.concat([pass_x, recp_x]))
    scatter_df.at[i, 'y'] = np.mean(np.concat([pass_y, recp_y]))
    scatter_df.at[i, 'number_passes'] = morocco_passes_df[morocco_passes_df.player_name == name].count().iloc[0]

scatter_df['marker_size'] = scatter_df['number_passes']/scatter_df['number_passes'].max()*1500
#counting passes between players

# Convert the column to strings
morocco_passes_df['player_name'] = morocco_passes_df['player_name'].astype(str)
morocco_passes_df['pass_recipient_name'] = morocco_passes_df['pass_recipient_name'].astype(str)

morocco_passes_df["pair_key"] = morocco_passes_df.apply(lambda x: "_".join(sorted([x["player_name"], x["pass_recipient_name"]])), axis=1)

lines_df = morocco_passes_df.groupby(["pair_key"]).x.count().reset_index()
lines_df.rename({'x': 'pass_count'}, axis='columns', inplace=True)

#setting a treshold. You can try to investigate how it changes when you change it.
lines_df = lines_df[lines_df['pass_count']>2]

pitch = Pitch(pitch_type="statsbomb", pitch_color='#22312b', line_color='#c7d5cc')
fig, axs = pitch.grid(figheight=10, title_height=0.08, endnote_space=0,axis=False, title_space=0, grid_height=0.82, endnote_height=0.05)
fig.set_facecolor('#22312b')
pitch.scatter(scatter_df.x, scatter_df.y, s=scatter_df.marker_size, ax=axs['pitch'], color='#c7d5cc',zorder = 3)

for i, row in scatter_df.iterrows():
    pitch.annotate(row.player_name, (row.x, row.y-3), ax=axs['pitch'], c='black', va='center', ha='center', weight = "bold", size=10, zorder = 4)

for i, row in lines_df.iterrows():
        player1 = row["pair_key"].split("_")[0]
        player2 = row['pair_key'].split("_")[1]
        if player1 == "nan" or player2 == "nan":
             continue
        #take the average location of players to plot a line between them
        player1_x = scatter_df.loc[scatter_df["player_name"] == player1]['x'].iloc[0]
        player1_y = scatter_df.loc[scatter_df["player_name"] == player1]['y'].iloc[0]
        player2_x = scatter_df.loc[scatter_df["player_name"] == player2]['x'].iloc[0]
        player2_y = scatter_df.loc[scatter_df["player_name"] == player2]['y'].iloc[0]
        num_passes = row["pass_count"]
        #adjust the line width so that the more passes, the wider the line
        line_width = (num_passes / lines_df['pass_count'].max() * 10)
        #plot lines on the pitch
        pitch.lines(player1_x, player1_y, player2_x, player2_y,
                        alpha=1, lw=line_width, zorder=2, color='#c7d5cc', ax = axs["pitch"])
# endnote /title
axs['endnote'].text(1, 0.5, '@arhrib_youssef', color='#c7d5cc',
                    va='center', ha='right', fontsize=15,
                    fontproperties=robotto_regular.prop)
fig.suptitle(f"Morocco  Passes Network vs {[team_name for team_name in team_names if team_name != 'Morocco'][0]}", fontsize = 20, color='#c7d5cc')
#plt.show()

#Centralisation
num_passes_df = morocco_passes_df.groupby(['player_name']).x.count().reset_index()
#print(num_passes_df.head())
num_passes_df.rename({'x': 'pass_count'}, axis='columns', inplace=True)
max_passes = num_passes_df['pass_count'].max()

denominator = 10*num_passes_df['pass_count'].sum()
nominator = (max_passes - num_passes_df['pass_count']).sum()

#calculate the centralisation index
centralisation_index = nominator/denominator
print("Centralisation index is ", centralisation_index)