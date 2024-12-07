import pandas as pd
from mplsoccer import Pitch, Sbopen, FontManager, VerticalPitch
from matplotlib import rcParams
import matplotlib.pyplot as plt
from highlight_text import ax_text
import numpy as np

pd.set_option("display.max_columns", None)
#rcParams['text.color'] = '#c7d5cc'  # set the default text color
fm_scada = FontManager('https://raw.githubusercontent.com/googlefonts/scada/main/fonts/ttf/'
                       'Scada-Regular.ttf')

# fontmanager for Google font (robotto)
robotto_regular = FontManager()
parser =Sbopen()
comp_df=parser.competition()

# arrow properties for the sub on/off
green_arrow = dict(arrowstyle='simple, head_width=0.7',
                   connectionstyle="arc3,rad=-0.8", fc="green", ec="green")
red_arrow = dict(arrowstyle='simple, head_width=0.7',
                 connectionstyle="arc3,rad=-0.8", fc="red", ec="red")

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
events_df, realted, freeze, tactics = parser.event(match_id=morocco_games_id[1])

#filter evenets by team name

team_names= events_df.team_name.unique()
print(events_df.counterpress.unique())
#Boolean mask for filtering the dataset by team and passes
morocco_passes_mask = (events_df['team_name']=='Morocco') & (events_df['type_name']=='Pass') & (events_df['sub_type_name'] != 'Throw-in')
morocco_passes_df  = events_df.loc[morocco_passes_mask, ['x', 'y', 'end_x', 'end_y', 'player_name','outcome_name', 'substitution_replacement_name', 'minute']]

morocco_Substitution_mask = (events_df['team_name']=='Morocco') & (events_df['type_name']=='Substitution') & (events_df['substitution_replacement_id'].notnull())
morocco_Substitution_df =  events_df.loc[morocco_Substitution_mask, [ 'player_name','substitution_replacement_id', 'substitution_replacement_name', 'minute']]
player_names = morocco_passes_df.player_name.unique()
#print(morocco_Substitution_df)
pitch = Pitch(pitch_type='statsbomb',  pitch_color="white", line_color='black')
fig,  axs = pitch.grid(nrows=4, ncols=4, grid_height=0.85, title_height=0.06,
                       axis=False,endnote_height=0.04,endnote_space=0.01,)
#fig.set_facecolor('#22312b')
for name, ax in zip(player_names, axs['pitch'].flat[:len(player_names)]):

        #player passes
    player_df = morocco_passes_df.loc[morocco_passes_df['player_name']==name]
        
    complete_passes = player_df[player_df.outcome_name.isnull()] 
    incomplete_passes = player_df[player_df.outcome_name.notnull()] 
    # print( len(player_df)  )
    if len(player_df) > 2 :
        #plot passes
        pitch.scatter(player_df.x, player_df.y, s=40, alpha=0.2,color='blue', ax=ax)
        pitch.arrows(incomplete_passes.x, incomplete_passes.y, incomplete_passes.end_x, 
                    incomplete_passes.end_y,color='#7065bb', width=2, 
                    headwidth=4, headlength=6, ax=ax,)
        pitch.arrows(complete_passes.x, complete_passes.y, complete_passes.end_x,
                    complete_passes.end_y,color='#56ae6c', width=2, 
                    headwidth=4, headlength=6, ax=ax,)
        # total passes
        annotation_string = (f'{name.split(' ')[-1] if len(name) > 20 else name} |'
                            f' <{len(complete_passes)}>/{len(player_df)} |'
                            f' {round(100*len(complete_passes)/len(player_df), 1)}%')
        ax_text(0, -5, annotation_string, ha='left', va='center', fontsize=10,
                fontproperties=fm_scada.prop, 
                highlight_textprops=[{"color": '#56ae6c'}], ax=ax)
            # add information for subsitutions on/off and arrows
        if name in morocco_Substitution_df['substitution_replacement_name'].values : 
            ax.text(104, -9, str(morocco_Substitution_df.loc[(morocco_Substitution_df['substitution_replacement_name']==name), 'minute' ].values[0]),
                fontsize=10,fontproperties=fm_scada.prop,ha='center', va='center')
            ax.annotate('', (108, -2), (100, -2), arrowprops=green_arrow)
        if name in morocco_Substitution_df['player_name'].values : 
            ax.text(116, -9, str(morocco_Substitution_df.loc[(morocco_Substitution_df['player_name']==name), 'minute' ].values[0]),
                fontsize=10,fontproperties=fm_scada.prop,ha='center', va='center')
            ax.annotate('', (120, -2), (112, -2), arrowprops=red_arrow)

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
