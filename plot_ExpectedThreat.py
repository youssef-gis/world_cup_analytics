#importing necessary libraries
import pandas as pd
import numpy as np
import json
# plotting
import matplotlib.pyplot as plt
#opening data
import os
import pathlib
import warnings
#used for plots
from mplsoccer import Pitch, FontManager
from scipy.stats import binned_statistic_2d

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')
robotto_regular = FontManager()
#Opening wyscout data
train_df = pd.DataFrame()
# for i in range(0,13):
#     URL="https://raw.githubusercontent.com/soccermatics/Soccermatics/refs/heads/main/course/lessons/data/Wyscout/events_England_"+str(i+1)+".json"
#     response = requests.get(URL)
#     response.raise_for_status()
#     json_data = response.json()
#     train_df = pd.concat([train_df, pd.DataFrame(json_data)] , ignore_index = True)

# train_df = train_df.loc[train_df.apply(lambda x: len(x.positions) == 2, axis=1)]
# train_df.to_csv(f"Outputs/plot_xGModelFit.csv", index=False)
train = pd.read_csv("Outputs/plot_xGModelFit.csv", delimiter=',')
next_event = train.shift(-1, fill_value=0)
train['next_event'] = next_event['subEventName']
train['kickedOut'] = train.apply(lambda x: 1 if x['next_event']=="Ball out of the field" else 0, axis=1)
train["positions"]= train["positions"].apply(lambda x: eval(x) )
train["tags"]= train["tags"].apply(lambda x: eval(x) )

#extract moving actions
move_df = train.loc[train['subEventName'].isin(['Simple pass', 'High pass', 'Head pass',
                                                 'Smart pass', 'Cross'])]
deleted_passes = move_df.loc[move_df['kickedOut']==1]
move_df = move_df.drop(deleted_passes.index)


#extract the coordinates
move_df['start_x'] = move_df.positions.apply(lambda x: (x[0]['x'])*105/100)
move_df['start_y'] = move_df.positions.apply(lambda x: (100-x[0]['y'])*68/100)
move_df['end_x'] = move_df.positions.apply(lambda x: (x[1]['x'])*105/100)
move_df['end_y'] = move_df.positions.apply(lambda x: (100-x[1]['y'])*68/100)
move_df= move_df.loc[(((move_df['end_x']!=0)&(move_df['end_y']!=68)) & 
                     ((move_df['end_x']!=105)&(move_df['end_y']!=0)))]

#Creating a pitch 2D hoistogram
pitch = Pitch(line_color='black', line_zorder=2, pitch_type='custom', pitch_length=105, 
                pitch_width=68)
move_count = pitch.bin_statistic(move_df.start_x, move_df.start_y, statistic='count',
                                  bins=(16, 12), normalize=False)
# fig, axs = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
#                     endnote_height=0.04, title_space=0, endnote_space=0)
# pcm = pitch.heatmap(move_count, ax=axs['pitch'], cmap='Blues', edgecolor='grey')

#Legend
# axs_cbar= fig.add_axes((0.83, 0.093, 0.03, 0.786))
# cbar = plt.colorbar(pcm, cax=axs_cbar)
# axs_endnote = axs['endnote'].text(1, 0.5, "@youssef_arhrib", va='center', ha='right',
#                                   fontproperties=robotto_regular.prop, fontsize=20,
#                                   color='black')

# axs_title = axs['title'].text(0.5, 0.1 ,'Moving actions 2D histogram', va='center', 
#                               ha='center',color='black',
#                               fontproperties=robotto_regular.prop, fontsize=20)
# fig.set_size_inches(10, 10)
# plt.show()
# fig.savefig(f"Outputs/plot_moving_actions.pdf", bbox_inches='tight', dpi=100)

moves_bins = move_count['statistic']

#Shots
shot_df = train.loc[train['subEventName']=='Shot']
shot_df['x'] = shot_df.positions.apply(lambda x: (x[0]['x'])*105/100)
shot_df['y'] = shot_df.positions.apply(lambda x: (100-x[0]['y'])*68/100)
#create 2D histogram of the shots
shot_count = pitch.bin_statistic(shot_df.x, shot_df.y, statistic='count', bins=(16, 12),
                                 normalize=False)

# fig, axs = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
#                      endnote_height=0.04, title_space=0, endnote_space=0)
# pcm = pitch.heatmap(shot_count, ax=axs['pitch'], cmap='Reds', edgecolor='grey')

# #Legend
# axs_cbar= fig.add_axes((0.83, 0.093, 0.03, 0.786))
# cbar = plt.colorbar(pcm, cax=axs_cbar)

# axs_endnote = axs['endnote'].text(1, 0.5, "@youssef_arhrib", va='center', ha='right',
#                                   fontproperties=robotto_regular.prop, fontsize=20,
#                                   color='black')

# axs_title = axs['title'].text(0.5, 0.1 ,'Shots 2D histogram', va='center', 
#                               ha='center',color='black',
#                               fontproperties=robotto_regular.prop, fontsize=20)
# fig.set_size_inches(10, 10)
# plt.show()
# fig.savefig(f"Outputs/plot_shots_actions.pdf", bbox_inches='tight', dpi=100)

shots_bins = shot_count['statistic']

#Goals
goal_df = shot_df.loc[shot_df.apply(lambda x: {'id': 101} in x.tags, axis=1)]
goals = pitch.bin_statistic(goal_df.x, goal_df.y, statistic='count', 
                            bins=(16, 12),normalize=False)
goals_bins = goals['statistic']

# fig, axs = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
#                      endnote_height=0.04, title_space=0, endnote_space=0)
# pcm = pitch.heatmap(goals, ax=axs['pitch'], cmap='Greens', edgecolor='grey')

# #Legend
# axs_cbar= fig.add_axes((0.83, 0.093, 0.03, 0.786))
# cbar = plt.colorbar(pcm, cax=axs_cbar)

# axs_endnote = axs['endnote'].text(1, 0.5, "@youssef_arhrib", va='center', ha='right',
#                                   fontproperties=robotto_regular.prop, fontsize=20,
#                                   color='black')

# axs_title = axs['title'].text(0.5, 0.1 ,'Goals 2D histogram', va='center', 
#                               ha='center',color='black',
#                               fontproperties=robotto_regular.prop, fontsize=20)
# fig.set_size_inches(10, 10)
# plt.show()
# fig.savefig(f"Outputs/plot_goals_actions.pdf", bbox_inches='tight', dpi=100)

#Move probability
move_prob = moves_bins / (moves_bins+shots_bins)
# fig, axs = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
#                       endnote_height=0.04, title_space=0, endnote_space=0)
# move_count['statistic'] = move_prob
# pcm = pitch.heatmap(move_count, ax=axs['pitch'], cmap='Blues', edgecolor='grey')
# #Legend
# axs_cbar= fig.add_axes((0.83, 0.093, 0.03, 0.786))
# cbar = plt.colorbar(pcm, cax=axs_cbar)

# axs_endnote = axs['endnote'].text(1, 0.5, "@youssef_arhrib", va='center', ha='right',
#                                   fontproperties=robotto_regular.prop, fontsize=20,
#                                   color='black')

# axs_title = axs['title'].text(0.5, 0.1 ,'Move probability 2D histogram', va='center', 
#                               ha='center',color='black',
#                               fontproperties=robotto_regular.prop, fontsize=20)
# fig.set_size_inches(10, 10)
# plt.show()
# fig.savefig(f"Outputs/plot_move_prob.pdf", bbox_inches='tight', dpi=100)


# #Shot probability
shot_prob = shots_bins / (moves_bins+shots_bins)
# fig, axs = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
#                       endnote_height=0.04, title_space=0, endnote_space=0)
# shot_count['statistic'] = shot_prob
# pcm = pitch.heatmap(shot_count, ax=axs['pitch'], cmap='Blues', edgecolor='grey')
# #Legend
# axs_cbar= fig.add_axes((0.83, 0.093, 0.03, 0.786))
# cbar = plt.colorbar(pcm, cax=axs_cbar)

# axs_endnote = axs['endnote'].text(1, 0.5, "@youssef_arhrib", va='center', ha='right',
#                                   fontproperties=robotto_regular.prop, fontsize=20,
#                                   color='black')

# axs_title = axs['title'].text(0.5, 0.1 ,'Shot probability 2D histogram', va='center', 
#                               ha='center',color='black',
#                               fontproperties=robotto_regular.prop, fontsize=20)
# fig.set_size_inches(10, 10)
# plt.show()
# fig.savefig(f"Outputs/plot_shot_prob.pdf", bbox_inches='tight', dpi=100)

#Goal probability
goal_prob = goals_bins / shots_bins
goal_prob[np.isnan(goal_prob)] = 0

#plotting it
# fig, axs = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
#                      endnote_height=0.04, title_space=0, endnote_space=0)
# goals['statistic'] = goal_prob
# pcm = pitch.heatmap(goals, ax=axs['pitch'], cmap='Blues', edgecolor='grey')

# #Legend
# axs_cbar= fig.add_axes((0.83, 0.093, 0.03, 0.786))
# cbar = plt.colorbar(pcm, cax=axs_cbar)

# axs_endnote = axs['endnote'].text(1, 0.5, "@youssef_arhrib", va='center', ha='right',
#                                   fontproperties=robotto_regular.prop, fontsize=20,
#                                   color='black')

# axs_title = axs['title'].text(0.5, 0.1 ,'Goal probability 2D histogram', va='center', 
#                               ha='center',color='black',
#                               fontproperties=robotto_regular.prop, fontsize=20)
# fig.set_size_inches(10, 10)
# plt.show()
# fig.savefig(f"Outputs/plot_goals_prob.pdf", bbox_inches='tight', dpi=100)

#Transition matrix
#move start index - using the same function as mplsoccer, it should work
move_df['start_sector'] = move_df.apply(
            lambda row: tuple([i[0] for i in binned_statistic_2d(np.ravel(row.start_x),
                                            np.ravel(row.start_y),values="None", statistic="count",
                                            bins=(16,12),range=[[0, 105], [0, 68]],
                                            expand_binnumbers=True)[3]]), axis=1)
#move end index
move_df['end_sector'] = move_df.apply(
            lambda row: tuple([i[0] for i in binned_statistic_2d(np.ravel(row.end_x),
                                            np.ravel(row.end_y),values="None", statistic="count",
                                            bins=(16,12),range=[[0, 105], [0, 68]],
                                            expand_binnumbers=True)[3]]), axis=1)
#count starting sector
df_count_starts = move_df.groupby(['start_sector'])['eventId'].count().reset_index()
df_count_starts.rename(columns={'eventId': 'count_starts'}, inplace=True)

transition_matrices = []
for i, row in df_count_starts.iterrows():
    start_sector = row['start_sector']
    count_starts = row['count_starts']
    #get all events thatstarted in this sector
    this_sector = move_df.loc[move_df['start_sector']==start_sector]
    df_count_ends = this_sector.groupby(['end_sector'])['eventId'].count().reset_index()
    df_count_ends.rename(columns={'eventId': 'count_ends'}, inplace=True)
    t_matrix = np.zeros((12, 16))
    for j, row2 in df_count_ends.iterrows():
        end_sector = row2['end_sector']
        value = row2['count_ends']
        t_matrix[end_sector[1]-1][end_sector[0]-1]=value
    t_matrix = t_matrix/count_starts
    transition_matrices.append(t_matrix)

#let's plot it for the zone [1,1] - left down corner
# fig, axs = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
#                      endnote_height=0.04, title_space=0, endnote_space=0)

#Change the index here to change the zone.

# goals["statistic"] = transition_matrices[90]
# pcm  = pitch.heatmap(goals, cmap='Reds', edgecolor='grey', ax=axs['pitch'])
#legend to our plot

# axs_cbar= fig.add_axes((0.83, 0.093, 0.03, 0.786))
# cbar = plt.colorbar(pcm, cax=axs_cbar)

# axs_endnote = axs['endnote'].text(1, 0.5, "@youssef_arhrib", va='center', ha='right',
#                                   fontproperties=robotto_regular.prop, fontsize=20,
#                                   color='black')

# axs_title = axs['title'].text(0.5, 0.1 ,'Transition probability for one of the middle zones', va='center', 
#                               ha='center',color='black',
#                               fontproperties=robotto_regular.prop, fontsize=20)
# fig.set_size_inches(10, 10)
# plt.show()
# fig.savefig(f"Outputs/plot_Transition_probability.pdf", bbox_inches='tight', dpi=100)

# plt.show()

#calculating Expected Threat
transition_matrices_array = np.array(transition_matrices)
xT = np.zeros((12, 16))
for i in range(5):
    shoot_expected_payoff= goal_prob*shot_prob
    move_expected_payoff= move_prob*(np.sum(np.sum(transition_matrices_array*xT, axis=2), 
                                                    axis=1).reshape(16, 12).T)
    xT = shoot_expected_payoff+move_expected_payoff
    #Let's plot it
    fig, axs = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
    goals['statistic']= xT
    # pcm = pitch.heatmap(goals, cmap= 'Oranges', edgecolor='grey', ax=axs['pitch'])
    # labels = pitch.label_heatmap(goals, color='blue', fontsize=8, ax=axs['pitch'],
    #                             ha='center', va='center', str_format="{0:,.2f}", zorder=3)
    # #Legend
    # axs_cbar= fig.add_axes((0.83, 0.093, 0.03, 0.786))
    # cbar = plt.colorbar(pcm, cax=axs_cbar)
    # txt = 'Expected Threat matrix after ' +  str(i+1) + ' moves'
    # axs_title = axs['title'].text(0.5, 0.1 ,txt, va='center', 
    #                               ha='center',color='black',
    #                               fontproperties=robotto_regular.prop, fontsize=20)
    # fig.set_size_inches(10, 10)
    # plt.show()

#add xT to move_df
successful_moves_df = move_df.loc[move_df.apply(lambda x: {'id': 1801} in x.tags, axis=1)]

#calculate xt 
successful_moves_df['xT_added'] =  successful_moves_df.apply(lambda row: xT[row.end_sector[1]-1][row.end_sector[0]-1]
                                                            - xT[row.start_sector[1]-1][row.start_sector[0]-1], axis=1)
#only progressives
value_adding_actions = successful_moves_df.loc[successful_moves_df['xT_added']>0]

#groupbyplayer
xT_by_player = value_adding_actions.groupby(["playerId"])['xT_added'].sum().reset_index()