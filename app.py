from flask import Flask, render_template
import pandas as pd
import base64
from io import BytesIO
import matplotlib.pyplot as plt

app = Flask(__name__)

def get_top_agents_graph(df):
    df['win'] = df['outcome'].apply(lambda x: 1 if x == 'Win' else 0)
    df['loss'] = df['outcome'].apply(lambda x: 1 if x == 'Loss' else 0)
    agent_stats = df.groupby('agent').agg({'win': 'sum', 'loss': 'sum', 'kdr': 'mean'})
    agent_stats['games_played'] = agent_stats['win'] + agent_stats['loss']
    agent_stats['win_loss_ratio'] = agent_stats['win'] / agent_stats['loss']
    agent_stats = agent_stats[agent_stats['games_played'] >= 20]
    top_agents = agent_stats.sort_values(by='win_loss_ratio', ascending=False).head(3)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(top_agents.index, top_agents['win_loss_ratio'], color='skyblue')
    plt.xlabel('Agent')
    plt.ylabel('Win/Loss Ratio')
    plt.title('Top 3 Agents by Win/Loss Ratio (Minimum 20 Games Played)')
    for bar, kdr in zip(bars, top_agents['kdr']):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'KDR: {round(kdr, 2)}', ha='center', va='bottom')
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png).decode('utf-8')
    buffer.close()
    plt.close()
    return graph

def get_top_maps_graph(df):
    df['win'] = df['outcome'].apply(lambda x: 1 if x == 'Win' else 0)
    df['loss'] = df['outcome'].apply(lambda x: 1 if x == 'Loss' else 0)
    map_stats = df.groupby('map').agg({'win': 'sum', 'loss': 'sum', 'kdr': 'mean'})
    map_stats['games_played'] = map_stats['win'] + map_stats['loss']
    map_stats['win_loss_ratio'] = map_stats['win'] / map_stats['loss']
    map_stats = map_stats[map_stats['games_played'] >= 20]
    top_maps = map_stats.sort_values(by='win_loss_ratio', ascending=False).head(3)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(top_maps.index, top_maps['win_loss_ratio'], color='lightcoral')
    plt.xlabel('Map')
    plt.ylabel('Win/Loss Ratio')
    plt.title('Top 3 Maps by Win/Loss Ratio (Minimum 20 Games Played)')
    for bar, kdr in zip(bars, top_maps['kdr']):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'KDR: {round(kdr, 2)}', ha='center', va='bottom')
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png).decode('utf-8')
    buffer.close()
    plt.close()
    return graph

def get_timeseries_graph(df, date_column, value_column, title):
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=date_column)
    plt.figure(figsize=(10, 6))
    plt.plot(df[date_column], df[value_column], marker='o', linestyle='-', color='b')
    plt.xlabel('Date')
    plt.ylabel(value_column)
    plt.title(title)
    plt.grid(True)
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png).decode('utf-8')
    buffer.close()
    plt.close()
    return graph

@app.route('/')
def index():
    df = pd.read_csv('valorant_games.csv')
    agent_graph = get_top_agents_graph(df)
    map_graph = get_top_maps_graph(df)
    timeseries_graph = get_timeseries_graph(df, 'date', 'kdr', 'KDR Over Time')
    return render_template('index.html', agent_graph=agent_graph, map_graph=map_graph, timeseries_graph=timeseries_graph)

if __name__ == '__main__':
    app.run(debug=True)