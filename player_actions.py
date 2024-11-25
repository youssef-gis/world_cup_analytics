from matplotlib.colors import to_rgba
import pandas as pd
from mplsoccer import Pitch, Sbopen, FontManager, VerticalPitch
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as path_effects
pd.set_option("display.max_columns", None)
rcParams['text.color'] = '#c7d5cc'  # set the default text color

# fontmanager for Google font (robotto)
robotto_regular = FontManager()
# path effects
path_eff = [path_effects.Stroke(linewidth=1.5, foreground='black'),
            path_effects.Normal()]
flamingo_cmap = LinearSegmentedColormap.from_list("Flamingo - 100 colors",
                                                  ['#e3aca7', '#c03a1d'], N=100)

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
events_df, realted, freeze, players = parser.event(match_id=morocco_games_id[3])

#filter events by team name
team_names= events_df.team_name.unique()
morocco_df = events_df[events_df.team_name == "Morocco"]
players_names= morocco_df.player_name.unique()[1: ]
player_name=  players_names[8]


pitch = Pitch(pitch_color='#22312b', line_color='#c7d5cc', line_zorder=2)
fig,  axs = pitch.grid(nrows=4, ncols=4, grid_height=0.85, title_height=0.06,
                       axis=False,endnote_height=0.04,endnote_space=0.01,)
fig.set_facecolor('#22312b')

for name, ax in zip(players_names, axs['pitch'].flat[:len(players_names)]):
    #player name
    ax.text(60, -5, name, ha='center', va='center', fontsize=10, color='#c7d5cc')
    player_events_df = morocco_df.loc[morocco_df.player_name == name, ['x', 'y']]
    kde = pitch.kdeplot(player_events_df.x, player_events_df.y, cmap='Blues', ax=ax,
                        fill=True, levels=100,
                        thresh=0,
                        cut=4, 
                        )
axs_title = axs['title'].text(0.5, 0.5 ,f'Morocco players Actions vs {[team_name for team_name in team_names if team_name != "Morocco"][0] }', va='center', 
                              ha='center',color='#c7d5cc',
                              fontproperties=robotto_regular.prop, fontsize=20)
axs_endnote = axs['endnote'].text(1, 0.5, "@youssef_arhrib", va='center', ha='right',
                                  fontproperties=robotto_regular.prop, fontsize=20,
                                  color='#c7d5cc')
plt.show()
fig.savefig(f'Outputs/Morocco_players_actions_vs_{[team_name for team_name in team_names if team_name != "Morocco"][0] }.pdf', dpi=100, bbox_inches=None)
