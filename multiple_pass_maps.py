import pandas as pd
from mplsoccer import Pitch, Sbopen, FontManager, VerticalPitch
from matplotlib import rcParams
import matplotlib.pyplot as plt

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
events_df, realted, freeze, tactics = parser.event(match_id=morocco_games_id[2])

#filter evenets by team name

team_names= events_df.team_name.unique()
#print(team_names)
#Boolean mask for filtering the dataset by team and passes
morocco_passes_mask = (events_df['team_name']=='Morocco') & (events_df['type_name']=='Pass') & (events_df['sub_type_name'] != 'Throw-in')
morocco_passes_df  = events_df.loc[morocco_passes_mask, ['x', 'y', 'end_x', 'end_y', 'player_name','outcome_name']]
player_names = morocco_passes_df.player_name.unique()

pitch = Pitch(pitch_type='statsbomb',  pitch_color="white", line_color='black')
fig,  axs = pitch.grid(nrows=4, ncols=4, grid_height=0.85, title_height=0.06,
                       axis=False,endnote_height=0.04,endnote_space=0.01,)
#fig.set_facecolor('#22312b')
for name, ax in zip(player_names, axs['pitch'].flat[:len(player_names)]):
    #player name
    ax.text(60, -5, name, ha='center', va='center', fontsize=10, color='black')

    #player passes
    player_df = morocco_passes_df.loc[morocco_passes_df['player_name']==name]
    #plot passes
    pitch.scatter(player_df.x, player_df.y, s=40, alpha=0.2,color='blue', ax=ax)
    pitch.arrows(player_df.x, player_df.y, player_df.end_x, player_df.end_y,
                  color='blue', width=1, ax=ax)

# for ax in axs['pitch'][-1, 16 - len(player_names):]:
#     ax.remove()

axs['title'].text(0.5, 0.5,
                  f'Morocco players passes against {[team_name for team_name in team_names if team_name !="Morocco"][0]}',
                    ha='center', va='center', fontsize=25, color="black")
axs_endnote = axs['endnote'].text(1, 0.5, "@youssef_arhrib", va='center', ha='right',
                                  fontproperties=robotto_regular.prop, fontsize=20,
                                  color='black')
plt.show()
fig.savefig(f'Outputs/Morocco_players_passes_vs_{[team_name for team_name in team_names if team_name != "Morocco"][0] }.pdf', dpi=100, bbox_inches=None)
