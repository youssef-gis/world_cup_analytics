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
morocco_passes_mask = (events_df.type_name == 'Pass') &  (events_df.team_name == "Morocco")
morocco_passes = events_df[morocco_passes_mask]

morocco_shots_mask = (events_df.type_name == 'Shot') &  (events_df.team_name == "Morocco")
morocco_shots = events_df[morocco_shots_mask]

passes_times = morocco_passes['minute']*60 + morocco_passes.second
shots_times = morocco_shots['minute']*60 + morocco_shots.second

def in_range(start, between, finish): 
    return (True in ((start<between) & (between<finish)).unique())

window_shot = 15
shots_start = shots_times - window_shot

passes_to_shots = passes_times.apply(lambda x: in_range(shots_start, x, shots_times))

is_corner = [x[1].sub_type_name == "Corner" for x in morocco_passes.iterrows()]

danger_passes_df = morocco_passes[np.logical_and(passes_to_shots, np.logical_not(is_corner))]

#count danger passes
count_passes = danger_passes_df.groupby(['player_name']).x.count()
print(count_passes)
pitch = Pitch(pitch_type="statsbomb", pitch_color='#22312b', line_color='#c7d5cc', line_zorder=2)
bins=(5, 5)

fig, axs = pitch.grid(figheight=10, title_height=0.08, endnote_space=0,axis=False, title_space=0, grid_height=0.82, endnote_height=0.05)
fig.set_facecolor('#22312b')

bin_stats = pitch.bin_statistic(danger_passes_df.x, danger_passes_df.y, statistic='count', bins=bins, normalize=False)
htmp = pitch.heatmap(bin_stats, ax=axs['pitch'], cmap='Reds', edgecolor='#f9f9f9')

#scatter the location on the pitch
pitch.scatter(danger_passes_df.x, danger_passes_df.y, s=100, color='blue', edgecolors='grey', linewidth=1, alpha=0.6, ax=axs["pitch"])


axs_title = axs['title'].text(0.5, 0.5 ,f'Morocco danger passes vs {[team_name for team_name in team_names if team_name != "Morocco"][0] }', va='center', 
                              ha='center',color='#c7d5cc',
                              fontproperties=robotto_regular.prop, fontsize=20)
axs_endnote = axs['endnote'].text(1, 0.5, "@youssef_arhrib", va='center', ha='right',
                                  fontproperties=robotto_regular.prop, fontsize=20,
                                  color='#c7d5cc')
plt.show()
fig.savefig(f'Outputs/Morocco_danger_passes_vs_{[team_name for team_name in team_names if team_name != "Morocco"][0] }.pdf', dpi=100, bbox_inches=None)

ax1 = count_passes.plot.bar(count_passes)
ax1.set_xlabel('')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)  # 45 degrees rotation
ax1.set_ylabel('Number of danger passes')
ax1.set_title('Danger passes by each player', color='black')
plt.show()