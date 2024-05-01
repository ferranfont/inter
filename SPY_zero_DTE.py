#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Zero DTE, entrada números redondos en el SPY el 0- DTE con un call o bull spread


# In[2]:


# Intraday Gamma Scalping SPY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import ivolatility as ivol
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# venta de put
fichero = 'SPY_2024-04-15_0_0_P.csv'
df_full = pd.read_csv(f'input_zero_DTE/{fichero}', sep=',', parse_dates=['timestamp'])

# Select the desired columns to keep
columns_to_keep = ['timestamp', 'optionStrike', 'optionType', 'optionBidPrice', 'optionAskPrice', 
                   'underlyingPrice', 'optionIv', 'optionDelta', 'optionTheta']
df_P_ATM = df_full[columns_to_keep]

# Extract the date part from the first row's timestamp
fecha = df_P_ATM['timestamp'].iloc[0].date()

print(fecha)
df_full


# Transposición del Dataframe según Strikes

# In[ ]:





# In[6]:


import pandas as pd

# Assuming df_P_ATM has already been loaded and is available

# Pivot the dataframe for bid, ask prices, optionIv, and optionTheta for each strike
prices_iv_theta_spreads = df_P_ATM.pivot_table(index='timestamp', 
                                               columns='optionStrike', 
                                               values=['optionBidPrice', 'optionAskPrice', 'optionIv', 'optionTheta'],
                                               aggfunc='first')

# Simplify the multi-index in columns
prices_iv_theta_spreads.columns = ['_'.join([str(col[0]), str(int(col[1]))]) for col in prices_iv_theta_spreads.columns.values]

# Get underlyingPrice without repeating for each strike
underlying_price = df_P_ATM[['timestamp', 'underlyingPrice']].drop_duplicates('timestamp').set_index('timestamp')

# Combine the dataframes
df_P_spread = pd.concat([prices_iv_theta_spreads, underlying_price], axis=1)

# Reset index to bring timestamp back as a column
df_P_spread.reset_index(inplace=True)

# Round all numeric columns to 2 decimal places
numeric_cols = df_P_spread.select_dtypes(include=['float64', 'int']).columns
df_P_spread[numeric_cols] = df_P_spread[numeric_cols].round(2)

# Dynamically calculate the Put_Spread based on available optionBidPrice and optionAskPrice columns
bid_prices = [col for col in df_P_spread.columns if 'optionBidPrice' in col]
ask_prices = [col for col in df_P_spread.columns if 'optionAskPrice' in col]

if bid_prices and ask_prices:
    # Sort to get the maximum bid price and minimum ask price columns
    highest_bid = max(bid_prices, key=lambda x: int(x.split('_')[-1]))
    lowest_ask = min(ask_prices, key=lambda x: int(x.split('_')[-1]))

    # Check if specific columns exist before trying to use them
    if highest_bid in df_P_spread and lowest_ask in df_P_spread:
        df_P_spread['Put_Spread'] = df_P_spread[highest_bid] - df_P_spread[lowest_ask]

# Optionally reorder columns or further process df_P_spread as required
df_P_spread.tail(30)


# In[ ]:





# Graph the results

# In[5]:


# Fetch the date from the first timestamp for use in the plot title
fecha = df_P_spread['timestamp'].iloc[0].strftime('%Y-%m-%d')

# Create a figure with specified figure size and three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1, 1]})

# Plotting Put_Spread and Underlying Price on the first subplot (ax1)
if 'Put_Spread' in df_P_spread.columns:
    ax1.plot(df_P_spread['timestamp'], df_P_spread['Put_Spread'] * 100, label='Put Spread', color='blue')
ax1_right = ax1.twinx()  # Create a second y-axis for the underlying price on the first subplot
ax1_right.plot(df_P_spread['timestamp'], df_P_spread['underlyingPrice'], label='SPY price', color='purple', linestyle='-', alpha=0.4)
ax1_right.set_ylabel('Underlying Price', color='purple')
ax1_right.legend(loc='upper right')
ax1.set_ylabel('Beneficio Put Spread x100', color='blue')
ax1.set_title(f'Put Spread 0 DTE en fecha {fecha}')

# Adding grid only for the underlying price axis
ax1_right.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='grey')

# Finding columns dynamically for optionBidPrice and optionAskPrice
bid_prices = [col for col in df_P_spread.columns if 'optionBidPrice' in col]
ask_prices = [col for col in df_P_spread.columns if 'optionAskPrice' in col]

# Plotting the first available bid and ask prices on the second subplot (ax2)
if bid_prices and ask_prices:
    ax2.plot(df_P_spread['timestamp'], df_P_spread[bid_prices[0]], label=f'Bid Price {bid_prices[0].split("_")[-1]}', color='green')
    ax2.plot(df_P_spread['timestamp'], df_P_spread[ask_prices[0]], label=f'Ask Price {ask_prices[0].split("_")[-1]}', color='red')
ax2.set_ylabel('Option Prices')
ax2.set_title('Option Bid and Ask Prices')
ax2.legend(loc='upper left')

# Finding columns dynamically for optionTheta
theta_columns = [col for col in df_P_spread.columns if 'optionTheta' in col]

# Plotting Theta values dynamically on the third subplot (ax3)
for theta in theta_columns:
    ax3.plot(df_P_spread['timestamp'], df_P_spread[theta], label=f'Theta {theta.split("_")[-1]}', color='orange' if '510' in theta else 'cyan')
ax3.set_xlabel('Time')
ax3.set_ylabel('Option Theta')
ax3.set_title('Option Theta for Available Strikes')
ax3.legend(loc='upper left')

# Show the plot
plt.tight_layout()  # Adjust layout so that labels do not overlap
plt.show()


# In[31]:


import os
import subprocess

def notebook_to_script(notebook_name, repo_url):
    script_name = f"{notebook_name}.py"
    convert_command = f"jupyter nbconvert --to script {notebook_name}.ipynb"
    
    try:
        subprocess.run(convert_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to convert notebook:", e)
        return  # Exit if conversion fails

    if not os.path.exists('.git'):
        subprocess.run("git init", shell=True, check=True)
    
    # Check current remote URL
    existing_url = subprocess.run("git remote get-url origin", shell=True, capture_output=True, text=True)
    if existing_url.returncode == 0 and existing_url.stdout.strip() == repo_url:
        print("Remote 'origin' is already set to the correct URL.")
    else:
        if existing_url.returncode == 0:
            print("Remote 'origin' already exists, resetting to new URL")
            subprocess.run(f"git remote set-url origin {repo_url}", shell=True, check=True)
        else:
            print("Adding new remote 'origin'.")
            subprocess.run(f"git remote add origin {repo_url}", shell=True, check=True)

    subprocess.run(f"git add {script_name}", shell=True, check=True)
    
    commit_message = "Add script generated from Jupyter Notebook"
    subprocess.run(f'git commit -m "{commit_message}"', shell=True, check=True)
    
    # Change here to push to 'main' instead of 'master'
    try:
        subprocess.run("git push -u origin main", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to push to GitHub:", e.stderr.decode())

# Usage example commented out
notebook_to_script('SPY_zero_DTE', 'https://github.com/ferranfont/inter.git')


# In[ ]:


import os
import subprocess

def notebook_to_script(notebook_name, repo_url):
    script_name = f"{notebook_name}.py"
    convert_command = f"jupyter nbconvert --to script {notebook_name}.ipynb"
    
    try:
        subprocess.run(convert_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to convert notebook:", e)
        return  # Exit if conversion fails

    if not os.path.exists('.git'):
        subprocess.run("git init", shell=True, check=True)
    
    # Check current remote URL
    existing_url = subprocess.run("git remote get-url origin", shell=True, capture_output=True, text=True)
    if existing_url.returncode == 0 and existing_url.stdout.strip() == repo_url:
        print("Remote 'origin' is already set to the correct URL.")
    else:
        if existing_url.returncode == 0:
            print("Remote 'origin' already exists, resetting to new URL")
            subprocess.run(f"git remote set-url origin {repo_url}", shell=True, check=True)
        else:
            print("Adding new remote 'origin'.")
            subprocess.run(f"git remote add origin {repo_url}", shell=True, check=True)

    subprocess.run(f"git add {script_name}", shell=True, check=True)
    
    commit_message = "Add script generated from Jupyter Notebook"
    subprocess.run(f'git commit -m "{commit_message}"', shell=True, check=True)
    
    # Change here to push to 'main' instead of 'master'
    try:
        subprocess.run("git push -u origin main", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to push to GitHub:", e.stderr.decode())

# Usage example commented out
# notebook_to_script('SPY_zero_DTE', 'https://github.com/ferranfont/inter.git')


# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


#subir a github automaticamente
# ejecutar esto en el CRM
#> git config --global credential.helper cache
# Set the cache to timeout after 1 hour (3600 seconds); adjust as needed
#> git config --global credential.helper 'cache --timeout=3600'
import os
import subprocess

def notebook_to_script(notebook_name, repo_url):
    script_name = f"{notebook_name}.py"
    convert_command = f"jupyter nbconvert --to script {notebook_name}.ipynb"
    
    try:
        subprocess.run(convert_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to convert notebook:", e)
        return  # Exit if conversion fails

    if not os.path.exists('.git'):
        subprocess.run("git init", shell=True, check=True)
    
    # Check current remote URL
    existing_url = subprocess.run("git remote get-url origin", shell=True, capture_output=True, text=True)
    if existing_url.returncode == 0 and existing_url.stdout.strip() == repo_url:
        print("Remote 'origin' is already set to the correct URL.")
    else:
        if existing_url.returncode == 0:
            print("Remote 'origin' already exists, resetting to new URL")
            subprocess.run(f"git remote set-url origin {repo_url}", shell=True, check=True)
        else:
            print("Adding new remote 'origin'.")
            subprocess.run(f"git remote add origin {repo_url}", shell=True, check=True)

    subprocess.run(f"git add {script_name}", shell=True, check=True)
    
    commit_message = "Add script generated from Jupyter Notebook"
    subprocess.run(f'git commit -m "{commit_message}"', shell=True, check=True)
    
    try:
        subprocess.run("git push -u origin master", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to push to GitHub:", e.stderr.decode())

# Usage example commented out
# notebook_to_script('SPY_zero_DTE', 'https://github.com/ferranfont/inter.git')


# In[28]:


import os
import subprocess

def setup_git_and_push(notebook_name, repo_url):
    script_name = f"{notebook_name}.py"
    convert_command = f"jupyter nbconvert --to script {notebook_name}.ipynb"
    
    try:
        subprocess.run(convert_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to convert notebook:", e)
        return  # Exit if conversion fails

    if not os.path.exists('.git'):
        subprocess.run("git init", shell=True, check=True)
    
    # Check if the correct remote is set, if not set it or reset it
    set_remote_command = f"git remote add origin {repo_url} || git remote set-url origin {repo_url}"
    subprocess.run(set_remote_command, shell=True, check=True)

    # Add, commit, and push changes to the main branch
    subprocess.run(f"git add {script_name}", shell=True, check=True)
    subprocess.run('git commit -m "Add converted script"', shell=True, check=True)
    subprocess.run("git push -u origin main", shell=True, check=True)  # Push to 'main' instead of 'master'

# Example usage:
setup_git_and_push('my_notebook', 'https://github.com/ferranfont/inter')


# In[6]:





# In[2]:





# In[3]:




