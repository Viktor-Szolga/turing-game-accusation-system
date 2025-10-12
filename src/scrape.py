# -----------------------------------------------------------------------------------------------------------------------------------------
# This file is inluded to show how the data was scraped. It was written by Simon Schmid and will not run outside of the Turing Game server.
# -----------------------------------------------------------------------------------------------------------------------------------------
import os
import mysql.connector
from dotenv import load_dotenv
import pickle
from ipwhois import IPWhois
load_dotenv()

import requests


import sys

if len(sys.argv) != 2:
    print("Usage: python scrape.py <output_path>")
    sys.exit(1)

output_path = sys.argv[1]


def get_ip_location(ip):

    obj = IPWhois(ip)
    result = obj.lookup_rdap()
    network = result.get('network', {})
    asn = result.get('asn', 'N/A')
    asn_country_code = result.get('asn_country_code', 'N/A')
    asn_description = result.get('asn_description', 'N/A')
    network_name = network.get('name', 'N/A')


    return {"country_code":asn_country_code,"asn":asn,"asn_description":asn_description,"network_name":network_name}


def get_mysql_connection():
    env_vars = ["MYSQL_HOST", "MYSQL_USERNAME", "MYSQL_PASSWORD", "MYSQL_DB"]
    for var in env_vars:
        print(os.getenv(var))
        if not os.getenv(var) or os.getenv(var).strip() == "":
            print(f"The .env file is missing or is not configured correctly for {var}. Please check and try again.")
            sys.exit(1)
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USERNAME"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB"),
    )



query_games = f"""
        SELECT *
        FROM games
        """

query_game_messages = f"""
        SELECT *
        FROM gamemessages
        """

query_bots = f"""
        SELECT *
        FROM api_keys
        """

query_gameplayers = f"""
        SELECT *
        FROM gameplayers
        """

query_sessions = f"""
        SELECT *
        FROM sessions
        """

query_users = f"""
        SELECT *
        FROM users
        """


conn = get_mysql_connection()





cursor = conn.cursor(dictionary=True)
cursor.execute(query_sessions)
db_sessions = cursor.fetchall()

sessions = {}

for s in db_sessions:
    sessions[s["sessionID"]] = s

sessions

cursor = conn.cursor(dictionary=True)
cursor.execute(query_users)
db_users = cursor.fetchall()

users = {}
for u in db_users:
    users[u["userID"]] = u

cursor.execute(query_gameplayers)
db_gameplayers = cursor.fetchall()

colors = {"deeppurple400":"Purple",
          "yellow400":"Yellow",
          "red400":"Red",
          "blue400":"Blue"}

gameplayers = {}
for player in db_gameplayers:
    game_id = player["gameID"]
    pl = player["userID"]
    judgeTime = player["judgetime"]
    color_id = player["colorID"]
    color = colors[color_id]
    session_id = player["sessionID"]
    if session_id in sessions:
        ip = sessions[session_id]['ip']
    else:
        ip = None
    username = users[pl]["username"]
    score = users[pl]["score"]

    if game_id not in gameplayers:
        gameplayers[game_id] = {}

    
    gameplayers[game_id][color] = {}

    gameplayers[game_id][color]["decision_time"] = judgeTime
    
        
    # if game_id in gameplayers:
    #     gameplayers[game_id][color] = judgeTime
    # else:
    #     gameplayers[game_id] = {color:judgeTime}

    #print(session_id,ip)
    if ip is not None:
        loc_data = get_ip_location(ip)
        print(loc_data)
        if ip == "83.164.178.213":
            # print("username:",username)
            # print(loc_data)
            # print("="*100)
            gameplayers[game_id][color]["ars_festival"] = True
        else:
            gameplayers[game_id][color]["ars_festival"] = False

        gameplayers[game_id][color]["country"] = loc_data['country_code']
        gameplayers[game_id][color]["network_details"] = loc_data
    gameplayers[game_id][color]["username"] = username
    gameplayers[game_id][color]["score"] = score


        

    #print(session_id,ip,loc)

    #gameplayers[game_id]["location"] = loc




cursor.execute(query_bots)
db_bots = cursor.fetchall()

bots = {}

for bot in db_bots:
    api_key = bot['api_key']
    bot_name = bot['bot_name']
    bots[api_key] = bot_name


cursor.execute(query_games)
db_games = cursor.fetchall()
games = {}

for game in db_games:
    game_id = game['gameID']
    games[game_id] = game
    
    if game['botmodel'] is not None:
        games[game_id]["botname"] = bots[game['botmodel']]
    else:
        games[game_id]["botname"] = None
    games[game_id]["messages"] = []

    if game_id in gameplayers:
        games[game_id]["player_info"] = gameplayers[game_id]


cursor.execute(query_game_messages)
db_messages = cursor.fetchall()

for m in db_messages:
    game_id = m['gameID']

    if game_id in games:
        games[game_id]["messages"].append(m)


for k,g in games.items():
    if len(g['messages']) != 0:
        if g['messages'][0]['message'].startswith("LANGUAGE"):
            l = g['messages'][0]['message'].split(" ")[1]
            games[k]["language"] = l
        else:
            games[k]["language"] = None
    else:
        games[k]["language"] = None






with open(output_path, "wb") as f:
    pickle.dump(games, f)

conn.close()