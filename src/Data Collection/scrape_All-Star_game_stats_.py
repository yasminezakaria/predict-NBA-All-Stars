# import libraries
import urllib2
from bs4 import BeautifulSoup
import requests
import pandas as pd

# specify the url
quote_page = "https://www.basketball-reference.com/allstar/NBA_2019.html"

# Get column headers from each page
# Assigns a new list of column headers each time this is called
def get_column_headers(soup):
    headers = []
    for th in soup.find('tr').findAll('th'):
        #print th.getText()
        headers.append(th.getText())
    #print headers # this line was for a bug check
    # Assign global variable to headers gathered by function
    return headers
#column_headers = [th.getText() for th in soup.find('tr').findAll('th')]

# Function to get player data from each page
def get_player_data(soup):
    # Temporary list within function to store data
    temp_player_data = []

    data_rows = soup.findAll('tr')[1:] # skip first row
    for i in range(len(data_rows)): # loop through each table row
        player_row = [] # empty list for each player row
        for td in data_rows[i].findAll('td'):
            player_row.append(td.getText()) # append separate data points
        temp_player_data.append(player_row) # append player row data
    return temp_player_data


def scrape_page(url):
    r = requests.get(url) # get the url
    soup = BeautifulSoup(r.text, 'html.parser') # Create BS object

    # call function to get column headers
    column_headers = get_column_headers(soup)
    print(column_headers)

    # call function to get player data
    player_data = get_player_data(soup)
    print(player_data.remove())

    # input data to DataFrame
    # Skip first value of column headers, 'Rk'
    df = pd.DataFrame(player_data, columns = column_headers[1:])

    return df

scrape_page(quote_page)
