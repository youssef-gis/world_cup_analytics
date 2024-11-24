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
events_df, realted, freeze, players = parser.event(match_id=morocco_games_id[1])

#filter events by team name
team_names= events_df.team_name.unique()
morocco_df = events_df[events_df['team_name'] == 'Morocco']
morocco_df.loc[morocco_df.tactics_formation.notnull(), 'tactics_id'] = morocco_df.loc[morocco_df.tactics_formation.notnull(), 'id']
morocco_df[['tactics_id', 'tactics_formation']] = morocco_df.groupby('team_name')[[
    'tactics_id', 'tactics_formation']].ffill()
formation_dict = {1: 'GK', 2: 'RB', 3: 'RCB', 4: 'CB', 5: 'LCB', 6: 'LB', 7: 'RWB',
                  8: 'LWB', 9: 'RDM', 10: 'CDM', 11: 'LDM', 12: 'RM', 13: 'RCM',
                  14: 'CM', 15: 'LCM', 16: 'LM', 17: 'RW', 18: 'RAM', 19: 'CAM',
                  20: 'LAM', 21: 'LW', 22: 'RCF', 23: 'ST', 24: 'LCF', 25: 'SS'}

players['position_abreviations'] = players['position_id'].map(formation_dict)

sub = morocco_df.loc[morocco_df['type_name']=='Substitution', ['tactics_id', 'player_id',
                        'substitution_replacement_id','substitution_replacement_name']]
players_sub = players.merge(sub.rename({'tactics_id': 'id'}, axis='columns'), on=['id', 'player_id'],
                            how='inner', validate='1:1')

players_sub = (players_sub[['id', 'substitution_replacement_id', 'position_abreviations']]
               .rename({'substitution_replacement_id': 'player_id'}, axis='columns'))

players= pd.concat([players, players_sub])
players.rename({'id': 'tactics_id'}, axis='columns', inplace=True)
players = players[['tactics_id', 'player_id', 'position_abreviations']]

morocco_df = morocco_df.merge(players, on=['tactics_id', 'player_id'], how='left', 
                            validate='m:1')

morocco_df = morocco_df.merge(players.rename({'player_id':'pass_recipient_id'}, axis='columns'),
                        on=['tactics_id', 'pass_recipient_id'],how='left',validate="m:1",
                        suffixes=['', '_receipt'])

FORMATION = morocco_df.tactics_formation.unique()[0]

morocco_passes = morocco_df.loc[(morocco_df.type_name == 'Pass') & 
                    (morocco_df.position_abreviations_receipt.notnull()), 
                    ['id', 'position_abreviations', 'position_abreviations_receipt']].copy()

locations_form = morocco_df.loc[(morocco_df['type_name'].isin(['Pass', 'Ball Receipt'])), ['position_abreviations', 'x', 'y']] 

# average locations
locations_avgs = locations_form.groupby('position_abreviations').agg({'x':['mean'], 'y': ['mean', 'count']})

locations_avgs.columns = ['x', 'y', 'count']

# calculate the number of passes between each position (using min/ max so we get passes both ways)

morocco_passes['pos_max'] = morocco_passes[['position_abreviations', 'position_abreviations_receipt']].max(axis=1)
morocco_passes['pos_min'] = morocco_passes[['position_abreviations', 'position_abreviations_receipt']].min(axis=1)

passes_between = morocco_passes.groupby(['pos_min', 'pos_max']).id.count().reset_index()
passes_between.rename({'id': 'passes_count'}, axis='columns', inplace=True)

passes_between = passes_between.merge(locations_avgs, left_on='pos_min', right_index=True)
passes_between = passes_between.merge(locations_avgs, left_on='pos_max', right_index=True,
                                        suffixes=['', '_end'])

MAX_LINE_WIDTH = 18
MAX_MARKER_SIZE = 3000
passes_between['width'] = (passes_between.passes_count / passes_between.passes_count.max() *
                           MAX_LINE_WIDTH)
locations_avgs['marker_size'] = (locations_avgs['count']
                                / locations_avgs['count'].max() * MAX_MARKER_SIZE)

MIN_TRANSPARENCY = 0.3
color = np.array(to_rgba('white'))
color = np.tile(color, (len(passes_between), 1))
c_transparency = passes_between.passes_count / passes_between.passes_count.max()
c_transparency = (c_transparency * (1 - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
color[:, 3] = c_transparency

pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
fig, axs = pitch.grid(figheight=10, title_height=0.08, endnote_space=0,
                      # Turn off the endnote/title axis. I usually do this after
                      # I am happy with the chart layout and text placement
                      axis=False,
                      title_space=0, grid_height=0.82, endnote_height=0.05)
fig.set_facecolor("#22312b")
pass_lines = pitch.lines(passes_between.x, passes_between.y,
                         passes_between.x_end, passes_between.y_end, lw=passes_between.width,
                         color=color, zorder=1, ax=axs['pitch'])
pass_nodes = pitch.scatter(locations_avgs.x, locations_avgs.y,
                           s=locations_avgs.marker_size,
                           color='red', edgecolors='black', linewidth=1, alpha=1, ax=axs['pitch'])
for index, row in locations_avgs.iterrows():
    pitch.annotate(row.name, xy=(row.x, row.y), c='white', va='center',
                   ha='center', size=16, weight='bold', ax=axs['pitch'])

# Load a custom font.
URL = 'https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Regular.ttf'
robotto_regular = FontManager(URL)

# endnote /title
axs['endnote'].text(1, 0.5, '@arhrib_youssef', color='#c7d5cc',
                    va='center', ha='right', fontsize=15,
                    fontproperties=robotto_regular.prop)
TITLE_TEXT = f"Morocco  Passes Network vs {[team_name for team_name in team_names if team_name != 'Morocco'][0]}"
axs['title'].text(0.5, 0.7, TITLE_TEXT, color='#c7d5cc',
                  va='center', ha='center', fontproperties=robotto_regular.prop, fontsize=30)


plt.show()
fig.savefig(f'Outputs/{TITLE_TEXT}.pdf', dpi=100, bbox_inches=None)
