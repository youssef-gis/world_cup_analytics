import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
# importing the modules 
from selenium import webdriver 
from selenium.webdriver.common.by import By
# using webdriver for chrome browser 
driver = webdriver.Chrome() 
try:
    url="https://fbref.com/en/comps/12/2021-2022/stats/2021-2022-La-Liga-Stats"
    driver.get(url)
    # response = requests.get(url)
    # response.raise_for_status()

except requests.exceptions.RequestException as e:
    print(f"Error Fetching  the webpage: {e}")

try:
    #soup= BeautifulSoup(response.content, 'html.parser')
    #tables = soup.find_all("table", {'class': "stats_table"})
    tables = driver.find_element(By.ID, "stats_standard")
    if not tables: 
       print("Error: Could not find the expected table on the page.")
    
    #print(tables.text)
   
    table= tables.text.split('\n')
    table.remove(table[0])
    data_content= [t.split(' ') for t in table[1: ]]
    print("Before reduction: " +str(len(data_content)))
    column_table=table[0].split(' ')
    ages = []
    minutes = []
    names = []
    for data in data_content:
        if data == ['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', 'MP', 'Starts', 'Min', '90s', 'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR', 'xG', 'npxG', 'xAG', 'npxG+xAG', 'PrgC', 'PrgP', 'PrgR', 'Gls', 'Ast', 'G+A', 'G-PK', 'G+A-PK', 'xG', 'xAG', 'xG+xAG', 'npxG', 'npxG+xAG', 'Matches']:
            data_content.remove(data)
    for data in data_content:
        #print("data length: "+str(len(data)))
       
        if len(data)==41:
            
            ages.append(data[9])
            minutes.append(data[13])
            names.append(' '.join(data[1:4]))

        elif len(data)==40:
            ages.append(data[8])
            minutes.append(data[12])
            names.append(' '.join(data[1:3]))
        elif len(data)==39:
            ages.append(data[7])
            minutes.append(data[11])
            if len(data[2]) > 2:
                names.append(' '.join(data[1:3]))
            else:
                names.append(data[1])
            #print(data[1:3])
        else: 
            ages.append(data[6])
            minutes.append(data[10])
            names.append(data[1])
    # print("After reduction: "+str(len(data_content)))
    # print(names[30], ages[30],minutes[30])
    df = pd.DataFrame(data=zip(names,ages,minutes),columns=['names','ages','minutes'])
    #print(df)
except Exception as e :
    print(f"Error parsing the HTML or reading the table: {e}")

try:
   # Step 5: Clean up the DataFrame.
   # Flatten multi-level headers if they exist, converting non-strings to empty strings.
   #if isinstance(df.columns, pd.MultiIndex):
       # Flatten the column headers, filtering out any None or NaN values.
       #df.columns = [" ".join(filter(None, map(str, col))) for col in df.columns]
   # Reset the index to clean up the DataFrame.
   df = df.reset_index(drop=True)
except Exception as e : 
    print(f"Error cleaning dataframe : {e}")


if df is not None:
    print('Successfully Extraxted the table')
    print(df.info())
    #save the file as csv
    file_name="minutes_played_by_player"
    df.to_csv(f"Outputs/{file_name}.csv", index=False)
    print(f"Data saved to {file_name}.csv")

print(df.minutes)