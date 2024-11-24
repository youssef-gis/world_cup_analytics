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
events_df, realted, freeze, players = parser.event(match_id=morocco_games_id[1])

#filter events by team name
team_names= events_df.team_name.unique()

morocco_pressures_df = events_df.loc[(events_df.type_name == "Pressure") & (events_df.team_name=="Morocco"), ['x', 'y']]

pitch = VerticalPitch(pitch_type="statsbomb", pitch_color="#22312b", 
                      line_zorder=2, line_color="#c7d5cc")

fig, axs = pitch.grid(endnote_height=0.03,endnote_space=0, figheight=8, title_height=0.06,
                      title_space=0, grid_height=0.86, axis=False)
fig.set_facecolor("#22312b")
robotto_regular = FontManager()
bins=(5,5)
bin_stats= pitch.bin_statistic(morocco_pressures_df.x, morocco_pressures_df.y,
                statistic='count', bins=bins, normalize=True)
htmp=pitch.heatmap(bin_stats, ax=axs['pitch'], cmap='Greens', edgecolor='#f9f9f9')
labels = pitch.label_heatmap(bin_stats, str_format='{:.0%}', color='#dee6ea', path_effects=path_eff,
                              va='center', ha='center', ax=axs['pitch'])
axs_endnote = axs['endnote'].text(1, 0.5, "@youssef_arhrib", va='center', ha='right',
                                  fontproperties=robotto_regular.prop, fontsize=20,
                                  color='#dee6ea')

axs_title = axs['title'].text(0.5, 0.5 ,f'Morocco pressure  vs {[team_name for team_name in team_names if team_name != "Morocco"][0] }', va='center', 
                              ha='center',color='#dee6ea', path_effects=path_eff,
                              fontproperties=robotto_regular.prop, fontsize=20)
fig.set_size_inches(10, 7)
fig.savefig(f'Outputs/Morocco_pressure_vs_{[team_name for team_name in team_names if team_name !="Morocco"][0]}.pdf', dpi=100)
plt.show()