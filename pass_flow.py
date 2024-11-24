from matplotlib.colors import to_rgba
import pandas as pd
from mplsoccer import Pitch, Sbopen, FontManager, VerticalPitch
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
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
events_df, realted, freeze, players = parser.event(match_id=morocco_games_id[0])

#filter events by team name
team_names= events_df.team_name.unique()

morocco_mask_pass = (events_df['type_name']=='Pass') & (events_df['team_name']=='Morocco')
morocco_passes = events_df.loc[morocco_mask_pass, ['x', 'y', 'end_x', 'end_y', 'outcome_name']]
completed_passes = morocco_passes.outcome_name.isnull()

pitch = Pitch(pitch_type="statsbomb", pitch_color='#22312b', line_color='#c7d5cc', line_zorder=2)
bins=(6, 4)

fig, axs = pitch.grid(figheight=10, title_height=0.08, endnote_space=0,axis=False, title_space=0, grid_height=0.82, endnote_height=0.05)
fig.set_facecolor('#22312b')

# plot the pass flow map with a custom color map and the arrows scaled by the average pass length
# the longer the arrow the greater the average pass length in the cell
grey = LinearSegmentedColormap.from_list('custom cmap', ['#DADADA', 'black'])

bins_heatmap = pitch.bin_statistic(morocco_passes.x, morocco_passes.y, statistic='count', bins=bins)
htmp_pitch = pitch.heatmap(bins_heatmap, ax=axs['pitch'], cmap='Reds')
flow_map = pitch.flow(morocco_passes.x, morocco_passes.y, 
            morocco_passes.end_x, morocco_passes.end_y,
            bins=bins, cmap=grey , ax=axs['pitch'], arrow_type='scale', arrow_length=15)

axs_title = axs['title'].text(0.5, 0.5 ,f'Morocco flow passes vs {[team_name for team_name in team_names if team_name != "Morocco"][0] }', va='center', 
                              ha='center',color='#dee6ea',
                              fontproperties=robotto_regular.prop, fontsize=20)
axs_endnote = axs['endnote'].text(1, 0.5, "@youssef_arhrib", va='center', ha='right',
                                  fontproperties=robotto_regular.prop, fontsize=20,
                                  color='#dee6ea')
plt.show()
fig.savefig(f'Outputs/Morocco_flow_passes_vs_{[team_name for team_name in team_names if team_name != "Morocco"][0] }.pdf', dpi=100, bbox_inches=None)
