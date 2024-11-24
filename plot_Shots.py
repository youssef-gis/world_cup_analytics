import pandas as pd
from mplsoccer import Pitch, Sbopen, FontManager, VerticalPitch
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams['text.color'] = '#c7d5cc'  # set the default text color
pd.set_option("display.max_columns", None)
parser = Sbopen()
compts=parser.competition()

wc_games = parser.match(competition_id=43, season_id=106)

morocco_games_ids = [game['match_id'] for i, game in wc_games.iterrows() if ((game['home_team_name']=="Morocco") | (game['away_team_name']=="Morocco"))]

game_events, related, freezer, tactics = parser.event(match_id=morocco_games_ids[1])
team_names = game_events['team_name'].unique()
morocco_shots = game_events[(game_events['team_name']=='Morocco') & (game_events['type_name']=='Shot')].set_index('id')

pitchLength=120
pitchWidth=80

pitch= VerticalPitch(pitch_type='statsbomb', pitch_color="#22312b", line_color='#c7d5cc' , half=True)
fig, axs = pitch.grid(endnote_height=0.03,endnote_space=0, figheight=8, title_height=0.06,
                      title_space=0, grid_height=0.86, axis=False)
fig.set_facecolor("#22312b")
robotto_regular = FontManager()
for i, shot in morocco_shots.iterrows():
    
    x=shot['x']
    y=shot['y']
    goal = (shot['outcome_name']=='Goal')#&(shot['sub_type_name']!='Penalty')
    circleSize=2
    if goal: 
        shotCircle= plt.Circle((y, x), circleSize, color='red')
        plt.text(y+1, x-2, shot['player_name'])
    else:
        shotCircle= plt.Circle((y, x), circleSize, color='red')
        shotCircle.set_alpha(0.2)
    axs['pitch'].add_patch(shotCircle)

axs_endnote = axs['endnote'].text(1, 0.5, "@youssef_arhrib", va='center', ha='right',
                                  fontproperties=robotto_regular.prop, fontsize=20,
                                  color='#dee6ea')

axs_title = axs['title'].text(0.5, 0.5 ,f'Morocco passes vs {[team_name for team_name in team_names if team_name != "Morocco"][0] }', va='center', 
                              ha='center',color='#dee6ea',
                              fontproperties=robotto_regular.prop, fontsize=20)
fig.set_size_inches(10, 7)
fig.savefig(f'Outputs/Morocco_Shots_vs_{[team_name for team_name in team_names if team_name !="Morocco"][0]}.pdf', dpi=100)
plt.show()