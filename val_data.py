import pandas as pd
import random
from datetime import datetime, timedelta

# Define constants
agents = ['Cypher', 'KAY/O', 'Brimstone', 'Phoenix', 'Breach']
maps = ['Ascent', 'Icebox', 'Lotus', 'Haven', 'Pearl', 'Fracture', 'Split', 'Bind']
ranks = ['Iron 1', 'Iron 2', 'Iron 3', 'Bronze 1', 'Bronze 2', 'Bronze 3', 'Silver 1', 'Silver 2', 'Silver 3', 'Gold 1', 'Gold 2', 'Gold 3']
outcomes = ['Win', 'Loss', 'Draw']

# Generate random data
data = []
start_date = datetime(2023, 1, 1)
for game_id in range(1, 101):
    episode = random.randint(1, 6)
    act = random.randint(1, 3)
    rank = random.choice(ranks)
    date = start_date + timedelta(days=random.randint(0, 365))
    agent = random.choice(agents)
    map_name = random.choice(maps)
    outcome = random.choice(outcomes)
    round_wins = random.randint(0, 15)
    round_losses = random.randint(0, 15)
    kills = random.randint(0, 30)
    deaths = random.randint(0, 30)
    assists = random.randint(0, 20)
    kdr = round(kills / (deaths if deaths != 0 else 1), 2)
    avg_dmg_delta = random.randint(-100, 100)
    headshot_pct = random.randint(0, 100)
    avg_dmg = random.randint(0, 300)
    acs = random.randint(0, 300)
    num_frag = random.randint(0, 10)
    
    data.append([game_id, episode, act, rank, date.strftime('%m/%d/%Y'), agent, map_name, outcome, round_wins, round_losses, kills, deaths, assists, kdr, avg_dmg_delta, headshot_pct, avg_dmg, acs, num_frag])

# Create DataFrame
columns = ['game_id', 'episode', 'act', 'rank', 'date', 'agent', 'map', 'outcome', 'round_wins', 'round_losses', 'kills', 'deaths', 'assists', 'kdr', 'avg_dmg_delta', 'headshot_pct', 'avg_dmg', 'acs', 'num_frag']
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv('fabricated_valorant_games.csv', index=False)