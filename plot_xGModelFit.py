# importing necessary libraries
import pandas as pd
import numpy as np
import json
# plotting
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch, Pitch, FontManager
# statistical fitting of models
import statsmodels.api as sm
import statsmodels.formula.api as smf
#opening data
import os, requests
import pathlib
import warnings

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

#Opening wyscout data
# train_df = pd.DataFrame()
# for i in range(1,13):
#     URL="https://raw.githubusercontent.com/soccermatics/Soccermatics/refs/heads/main/course/lessons/data/Wyscout/events_England_"+str(i)+".json"
#     response = requests.get(URL)
#     response.raise_for_status()
#     json_data = response.json()
#     train_df = pd.concat([train_df, pd.DataFrame(json_data)])
#train_df.to_csv(f"Outputs/plot_xGModelFit.csv", index=False)
train_df = pd.read_csv("Outputs/plot_xGModelFit.csv", delimiter=",")
Shot_mask = (train_df.subEventName == 'Shot')
shot_df = train_df.loc[Shot_mask]
#get shot coordinates as separate columns 
shot_df['positions'] = shot_df.positions.apply(eval) 
shot_df['X'] = shot_df.positions.apply(lambda pos: (100 - pos[0]['x'])*105/100)
shot_df['y'] = shot_df.positions.apply(lambda pos: pos[0]['y'] * 68/100)
shot_df['C'] = shot_df['y'].apply(lambda y: abs(y - 50)*68/100)
shot_df['distance'] = np.sqrt(shot_df['X']**2 + shot_df['C']**2)
shot_df['Angle'] = np.where(np.arctan(7.32*shot_df["X"]/(shot_df["X"]**2 + shot_df["C"]**2 - (7.32/2)**2)) > 0,
                            np.arctan(7.32*shot_df["X"]/(shot_df["X"]**2+shot_df["C"]**2 - (7.32/2)**2)), 
                            np.arctan(7.32 * shot_df["X"] /(shot_df["X"]**2 + shot_df["C"]**2 - (7.32/2)**2)) + np.pi)

shot_df['tags'] = shot_df.tags.apply(eval) 

shot_df['Goal'] = shot_df.tags.apply(lambda shot: 1 if {'id':101} in shot else 0).astype(object)


# pitch = VerticalPitch(pitch_type='custom', half=True, label=False, pitch_width=68, pitch_length=105, line_zorder=2)
# fig, axs = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
#                      endnote_height=0.04, title_space=0, endnote_space=0)
# #nmbr of shots inn each bin
# bin_stats_shots = pitch.bin_statistic(105-shot_df['X'], shot_df['y'], statistic='count',
#                                        bins=50)

#plot goals locations
goals = shot_df.loc[shot_df['Goal']==1]

# #Goals bins stats
# bin_stats_goals = pitch.bin_statistic(105 - goals.X, goals.y, statistic='count', bins=50)

# #Plotting the probability of scoring a goal given the location
# bin_stats = pitch.bin_statistic(105 - shot_df.X, shot_df.y, bins=50)
# #normalize number of goals by number of shots for every bin
# bin_stats['statistic'] = bin_stats_goals['statistic']/bin_stats['statistic']

# #make heatmap
# pitch_map = pitch.heatmap(bin_stats, ax=axs['pitch'], cmap='Reds', edgecolor='white', 
#                           linewidth=0.01 )
# #make colorbar_legend
# ax_colorbar = fig.add_axes((0.90, 0.05, 0.04, 0.8))
# cbar = plt.colorbar(pitch_map, cax=ax_colorbar)

# robotto_regular = FontManager()
# axs_endnote = axs['endnote'].text(1, 0.5, "@youssef_arhrib", va='center', ha='right',
#                                   fontproperties=robotto_regular.prop, fontsize=20,
#                                   color='black')

# axs_title = axs['title'].text(0.5, 0.2 ,f'Probability of scoring map - 2017/2018 Premier League Season', va='center', 
#                               ha='center',color='black',
#                               fontproperties=robotto_regular.prop, fontsize=30)
# fig.set_size_inches(10, 7)
#fig.savefig('Outputs/Probability_scoring_map_2017_2018_Premier_League_Season.pdf', dpi=100)



##Plotting a logistic curve
# b=[3, -3]
# x=np.arange(5, step=0.1)
# y=1/(1+np.exp(b[0]+b[1]*x))
# fig, ax = plt.subplots()
# plt.xlim((0, 5))
# plt.ylim((-0.05, 1.05))
# ax.plot(x, y)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

##relationship between goals and angles
# shots_200 = shot_df.iloc[ :200]
# print(shots_200)
# fig, ax = plt.subplots()
# ax.plot(shots_200['Angle']*180/np.pi, shots_200['Goal'],
#         linestyle='none',  marker= '.',markersize=10)

# ax.set_xlabel("Shot Angles (degrees)")
# ax.set_ylabel('Goals')
# ax.set_ylim((-0.05, 1.05))
# ax.set_yticks([0, 1])
# ax.set_yticklabels(['No', 'Yes'])

## the relationship between probability of scoring goals and angle
##number of shots from an angle: 
shotcount = np.histogram(shot_df['Angle']*180/np.pi, bins=40, range=[0, 150])
##number of goals from an angle: 
goals_count = np.histogram(goals['Angle']*180/np.pi, bins=40, range=[0, 150])
np.seterr(divide='ignore', invalid='ignore')
prob_goal = np.divide(goals_count[0], shotcount[0])

angle= shotcount[1]

mid_angle = (angle[:-1]+angle[1:])/2
shot_df['Angle_degrees'] = shot_df['Angle'].apply(lambda x: x*180/np.pi)

# fig, axs = plt.subplots()
# axs.plot(mid_angle, prob_goal, linestyle='none', marker='.', markersize=10)
# axs.set_xlabel("shot Angles (degrees)")
# axs.set_ylabel('Probability of Shots Scored')
# axs.spines['top'].set_visible(False)
# axs.spines['right'].set_visible(False)

##Fitting logistic regression and finding best parameters
#test_model = smf.glm('Goal ~ Angle', data=shot_df, family= sm.families.Binomial()).fit()
#print(test_model)
#gets params
#b=test_model.params

#calculate xg
#xg_prob = 1/(1+np.exp(b[0]+b[1]*mid_angle*np.pi/180))
#plot_data
# fig, axs = plt.subplots()
# axs.plot(mid_angle, prob_goal, linestyle="none", color='black', marker='.', markersize=12)
# axs.plot(mid_angle, xg_prob, linestyle='solid', color='black')
# axs.set_xlabel("shot Angles (degrees)")
# axs.set_ylabel("Probability Shots scored")
# axs.spines['top'].set_visible(False)
# axs.spines['right'].set_visible(False)

#investigating relationship between probability of scoring and distance to goal
shots_distance = np.histogram(shot_df['distance'], bins=40, range=[0, 70])
goals_distance = np.histogram(goals['distance'], bins=40, range=[0, 70])

prob_goal_dist = np.divide(goals_distance[0], shots_distance[0])
dist = shots_distance[1]
mid_dist = (dist[:-1]+dist[1:])/2
# fig, axs = plt.subplots()
# axs.plot(mid_dist, prob_goal_dist, linestyle='none', marker='.', markersize=12)
# axs.set_xlabel('Shot distances (m)')
# axs.set_ylabel('Shots probability to be scored')
# axs.spines['top'].set_visible(False)
# axs.spines['right'].set_visible(False)

# train_model_by_dist = smf.glm('Goal ~ distance', data=shot_df, family= sm.families.Binomial()).fit()
# print(train_model_by_dist.summary())
# b=train_model_by_dist.params
# xg_prob_dist = 1/(1+np.exp(b[0]+b[1]*mid_dist))
# axs.plot(mid_dist, xg_prob_dist, linestyle='solid', color='black')

##Adding squared distance to the model
shot_df['distance_squared'] = shot_df['distance']**2
#test_model_dist = smf.glm('Goal ~ distance + distance_squared',data=shot_df, family=sm.families.Binomial()).fit()
#print(test_model_dist.summary())
#b= test_model_dist.params
#xgprob_dist_squared = 1/(1+np.exp(b[0]+b[1]*mid_dist+b[2]*pow(mid_dist, 2)))
#axs.plot(mid_dist, xgprob_dist_squared, linestyle='solid', color='black')

#creating extra variables
shot_df['X_squared']= shot_df['X']**2
shot_df['C_squared']= shot_df['C']**2
shot_df['AX']= shot_df['Angle']*shot_df['X']
# list the model variables you want here
model_variables = ['Angle', 'distance', 'X', 'C', 'X_squared', 'C_squared', 'AX']
model=''
for v in model_variables[:-1]:
    model = model + v + ' + '
model = model + model_variables[-1]

test_model = smf.glm('Goal ~ '+model, data=shot_df, family=sm.families.Binomial()).fit()
#print(test_model.summary())
b= test_model.params

#return xG value for more general model
def calc_xg(sh):
    bsum=b[0]
    for i, v in enumerate(model_variables):
        bsum = bsum + b[i+1]*sh[v]
    xG= 1/(1+np.exp(bsum))
    return xG
#add an xG to my dataframe
XG = shot_df.apply(calc_xg, axis=1)
shot_df = shot_df.assign(XG=XG)
#Create a 2D map of xG
pgoal_2d=np.zeros((68,68))
for x in range(68):
    for y in range(68):
        sh = dict()
        a = np.arctan(7.32*x/(x**2+abs(y-68/2)**2 - (7.32/2)**2 ))
        if a<0 :
            a=np.pi + a
        sh['Angle']=a
        sh['distance']=np.sqrt(x**2+abs(y-68/2)**2)
        sh['D_squared']=x**2+abs(y-68/2)**2
        sh['X']= x
        sh['X_squared'] = x**2
        sh['C'] = abs(y-68/2)
        sh['C_squared'] = abs(y-68/2)**2
        sh['AX']= x*a

        pgoal_2d[x, y]= calc_xg(sh)

# pitch = VerticalPitch(pitch_type='custom', pitch_length=105, pitch_width=68, line_color='black',
#                     line_zorder=2, half=True)
# fig, axs = pitch.draw()
# ##plot probability
# pos = axs.imshow(pgoal_2d, extent=[-1, 68, 68, 1], aspect='auto', 
#                     cmap=plt.cm.Reds,vmin=0, vmax=0.3, zorder = 1)
# fig.colorbar(pos, ax=axs)
# #make legend
# axs.set_title('Probability of goal')
# plt.xlim((0, 68))
# plt.ylim((0, 60))
# plt.gca().set_aspect('equal', adjustable='box')

#Testing model fit
# Mcfaddens Rsquared for Logistic regression
null_model = smf.glm(formula='Goal ~ 1', data=shot_df, family=sm.families.Binomial()).fit()
print("Mcfaddens Rsquared", 1 - test_model.llf / null_model.llf)

# ROC curve
numobs=100
TP = np.zeros(numobs)
FP = np.zeros(numobs)
TN = np.zeros(numobs)
FN = np.zeros(numobs)

for i, threshold in enumerate(np.arange(0, 1, 1/numobs)):
    for j, shot in shot_df.iterrows():
        if shot['Goal']==1:
            if shot['XG'] > threshold:
                TP[i] = TP[i]+1
            else: 
                FN[i] = FN[i] +1
        
        if shot['Goal']==0:
            if shot['XG'] > threshold:
                FP[i] = FP[i]+1
            else:
                TN[i]=TN[i]+1
            
fig, axs = plt.subplots()
axs.plot(FP/(FP+TN), TP/(TP+FN), color='black')
axs.plot([0, 1], [0, 1], linestyle='dotted', color='black')
axs.set_ylabel("Predicted to score and did TP/(TP+FN))")
axs.set_xlabel("Predicted to score but didn't FP/(FP+TN)")
plt.ylim((0.00, 1.00))
plt.xlim((0.00, 1.00))
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
plt.show()


