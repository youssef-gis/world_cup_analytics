import requests
#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from bs4 import BeautifulSoup
from io import StringIO
# importing the modules 
from selenium import webdriver 
from selenium.webdriver.common.by import By
# using webdriver for chrome browser 
# driver = webdriver.Chrome() 
# try:
#     url="https://fbref.com/en/comps/12/2021-2022/stats/2021-2022-La-Liga-Stats"
#     driver.get(url)
#     # response = requests.get(url)
#     # response.raise_for_status()

# except requests.exceptions.RequestException as e:
#     print(f"Error Fetching  the webpage: {e}")

# try:
#     #soup= BeautifulSoup(response.content, 'html.parser')
#     #tables = soup.find_all("table", {'class': "stats_table"})
#     tables = driver.find_element(By.ID, "stats_standard")
#     if not tables: 
#        print("Error: Could not find the expected table on the page.")
    
#     #print(tables.text)
   
#     table= tables.text.split('\n')
#     table.remove(table[0])
#     data_content= [t.split(' ') for t in table[1: ]]
#     print("Before reduction: " +str(len(data_content)))
#     column_table=table[0].split(' ')
#     ages = []
#     minutes = []
#     names = []
#     for data in data_content:
#         if data == ['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', 'MP', 'Starts', 'Min', '90s', 'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR', 'xG', 'npxG', 'xAG', 'npxG+xAG', 'PrgC', 'PrgP', 'PrgR', 'Gls', 'Ast', 'G+A', 'G-PK', 'G+A-PK', 'xG', 'xAG', 'xG+xAG', 'npxG', 'npxG+xAG', 'Matches']:
#             data_content.remove(data)
#     for data in data_content:
#         #print("data length: "+str(len(data)))
       
#         if len(data)==41:
            
#             ages.append(data[9])
#             minutes.append(data[13])
#             names.append(' '.join(data[1:4]))

#         elif len(data)==40:
#             ages.append(data[8])
#             minutes.append(data[12])
#             names.append(' '.join(data[1:3]))
#         elif len(data)==39:
#             ages.append(data[7])
#             minutes.append(data[11])
#             if len(data[2]) > 2:
#                 names.append(' '.join(data[1:3]))
#             else:
#                 names.append(data[1])
#             #print(data[1:3])
#         else: 
#             ages.append(data[6])
#             minutes.append(data[10])
#             names.append(data[1])
#     # print("After reduction: "+str(len(data_content)))
#     # print(names[30], ages[30],minutes[30])
#     df = pd.DataFrame(data=zip(names,ages,minutes),columns=['names','ages','minutes'])
#     #print(df)
# except Exception as e :
#     print(f"Error parsing the HTML or reading the table: {e}")

# try:
#    # Step 5: Clean up the DataFrame.
#    df = df.reset_index(drop=True)
# except Exception as e : 
#     print(f"Error cleaning dataframe : {e}")

# if df is not None:
#     print('Successfully Extraxted the table')
#     #print(df.info())
#     #save the file as csv
#     file_name="minutes_played_by_player"
#     df.to_csv(f"Outputs/{file_name}.csv", index=False)
#     print(f"Data saved to {file_name}.csv")

df = pd.read_csv('./Outputs/minutes_played_by_player.csv', delimiter=',')

num_observ = 200
minutes_ages_model = pd.DataFrame()
minutes_ages_model = minutes_ages_model.assign(names=df['names'][0: num_observ])
minutes_ages_model = minutes_ages_model.assign(ages=df['ages'][0: num_observ])
minutes_ages_model = minutes_ages_model.assign(minutes=df['minutes'][0: num_observ])

# Convert 'StringColumn' to float using astype()
# Use the `replace()` function to remove symbols
minutes_ages_model['minutes'] = minutes_ages_model['minutes'].replace(',', '', regex=True)

minutes_ages_model['minutes'] =minutes_ages_model['minutes'].astype(float)
minutes_ages_model['ages'] =minutes_ages_model['ages'].astype(int)


# Make an age squared column so we can fir polynomial model.
minutes_ages_model = minutes_ages_model.assign(
                                age_squared = np.power(minutes_ages_model['ages'], 2))
minutes_ages_model = minutes_ages_model.assign(
                                age_cubed = np.power(minutes_ages_model['ages'], 3))
#print(minutes_ages_model.head())
#Plotting the data
fig, axs = plt.subplots()
axs.plot(minutes_ages_model['ages'], minutes_ages_model['minutes'], 
            linestyle='none', marker='.', markersize=10, color='blue')

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)

axs.set_ylabel('Minutes played') 
axs.set_xlabel('Ages')

plt.xlim((15, 40))
plt.ylim((0, 3000))
   

#straight line relationship between minutes played and the age of player
model_fit = smf.ols('minutes ~ ages ', data=minutes_ages_model).fit()
#print(model_fit.summary())
b=model_fit.params
#A model including squared terms
model_fit_polynomial = smf.ols('minutes ~ ages + age_squared +age_cubed', data=minutes_ages_model).fit()
print(model_fit_polynomial.summary())
b1=model_fit_polynomial.params
print(b1.Intercept)
x= np.arange(40, step=1)
#y= np.mean(minutes_ages_model['minutes'])*np.ones(40) #straight line
y= b1.iloc[0] + b1.iloc[1]*x + b1.iloc[2]*x*x +b1.iloc[3]*x*x*x
axs.plot(x, y, color='black')

for i, age in enumerate(minutes_ages_model['ages']):
    #axs.plot([age, age], [minutes_ages_model['minutes'][i], np.mean(minutes_ages_model['minutes'])], color='red') #straight line
    axs.plot([age, age], 
            [minutes_ages_model['minutes'][i],b1.Intercept + b1.ages*age + 
            b1.age_squared*age*age +  b1.age_cubed*age*age*age],color='red')
plt.show()        