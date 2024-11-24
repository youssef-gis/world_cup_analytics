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
events_df, realted, freeze, tactics = parser.event(match_id=morocco_games_id[4])

#filter evenets by team name

team_names= events_df.team_name.unique()
print(team_names)
#Boolean mask for filtering the dataset by team and passes
morocco_passes_mask = (events_df['team_name']=='Morocco') & (events_df.pass_assisted_shot_id.notnull()) #(events_df['type_name']=='Pass')
morocco_passes_df  = events_df.loc[morocco_passes_mask, ['x', 'y', 'end_x', 'end_y', 'pass_assisted_shot_id']]
pass_completed = events_df.outcome_name.isnull()

#passes leading to shots
morocco_shot_mask = (events_df['team_name']=='Morocco') & (events_df['type_name']=='Shot')
morocco_shot_df  = (events_df.loc[morocco_shot_mask, ['id', 'x', 'y', 'outcome_name']].rename({'id': 'pass_assisted_shot_id'}, axis=1))

morocco_chances = morocco_passes_df.merge(morocco_shot_df, how='left', on='pass_assisted_shot_id', suffixes=('_passes', '_shots')).drop('pass_assisted_shot_id', axis=1)

morocco_mask_goal = morocco_chances.outcome_name == "Goal"

#Draw passes
pitch = Pitch(pitch_type='statsbomb', pitch_color="#22312b", line_color='#c7d5cc', )
fig, axs = pitch.grid(endnote_height=0.03,endnote_space=0, figheight=8, title_height=0.06,
                      title_space=0, grid_height=0.86, axis=False)
fig.set_facecolor('#22312b')

#Draw completed passes 
# pitch.arrows(morocco_passes_df[pass_completed].x, morocco_passes_df[pass_completed].y,
#               morocco_passes_df[pass_completed].end_x, morocco_passes_df[pass_completed].end_y,
#             ax=axs['pitch'], width=2, headwidth=10, headlength=10, color="blue", label="Completed Passes" )
# #Draw Other passes 
# pitch.arrows(morocco_passes_df[~pass_completed].x, morocco_passes_df[~pass_completed].y,
#             morocco_passes_df[~pass_completed].end_x, morocco_passes_df[~pass_completed].end_y,
#             ax=axs['pitch'], width=2, headwidth=10, headlength=10, color="red", label="Other Passes" )


#plot passes leading to shots  
pitch.lines(morocco_chances['x_passes'], morocco_chances['y_passes'], morocco_chances['end_x'], morocco_chances['end_y'], ax=axs['pitch'], 
            lw=10, transparent=True, comet=True, cmap='jet', label='pass leading to shot')

#plot_goals
pitch.scatter(morocco_chances[morocco_mask_goal].end_x, morocco_chances[morocco_mask_goal].end_y, ax=axs['pitch'], s=700,marker='football', zorder=2, label='goals', edgecolors='black')

#plot_shots
pitch.scatter(morocco_chances[~morocco_mask_goal].end_x, morocco_chances[~morocco_mask_goal].end_y, ax=axs['pitch'], s=700, color='#22312b', zorder=2, label='shots', edgecolors='white' )

legend = axs['pitch'].legend(facecolor='#22312b', edgecolor='None',
                             loc='lower center', handlelength=4)

for text in legend.get_texts():
    text.set_fontproperties(robotto_regular.prop)
    text.set_fontsize(20)

axs_title = axs['title'].text(0.5, 0.5 ,f'Morocco passes vs {[team_name for team_name in team_names if team_name != "Morocco"][0] }', va='center', 
                              ha='center',color='#dee6ea',
                              fontproperties=robotto_regular.prop, fontsize=20)
axs_endnote = axs['endnote'].text(1, 0.5, "@youssef_arhrib", va='center', ha='right',
                                  fontproperties=robotto_regular.prop, fontsize=20,
                                  color='#dee6ea')
plt.show()
fig.savefig(f'Outputs/Morocco_passes_leading_to_shots_vs_{[team_name for team_name in team_names if team_name != "Morocco"][0] }.pdf', dpi=100, bbox_inches=None)



