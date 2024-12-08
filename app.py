from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

def timeseries(df, validation_data, predictions):
    results_df = pd.DataFrame({
        'Game': validation_data.index,
        'Actual': validation_data['rank_numeric'],
        'Predicted': predictions.flatten()
    })

    mse = mean_squared_error(results_df['Actual'], results_df['Predicted'])
    mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])

    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-60:], df['rank_numeric'][-60:], label='Historical', color='blue')  # Show more history
    plt.plot(results_df['Game'], results_df['Predicted'], 
             label='Predicted', linestyle='--', color='red')
    plt.plot(results_df['Game'], results_df['Actual'], 
             label='Actual', color='green')

    plt.title(f'Valorant Rank 30 Day Prediction\nMSE: {mse:.2f}, MAE: {mae:.2f}')
    plt.xlabel('Game Number')
    plt.ylabel('Rank')
    plt.legend()
    plt.tight_layout()
    # Save plot to buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png).decode('utf-8')
    buffer.close()

    # Clear the current figure
    plt.close()

    return graph

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def process_data(df):
    rank_mapping = {
        'Iron 1': 1, 'Iron 2': 2, 'Iron 3': 3,
        'Bronze 1': 4, 'Bronze 2': 5, 'Bronze 3': 6,
        'Silver 1': 7, 'Silver 2': 8, 'Silver 3': 9,
        'Gold 1': 10, 'Gold 2': 11, 'Gold 3': 12,
        'Platinum 1': 13, 'Platinum 2': 14, 'Platinum 3': 15,
        'Diamond 1': 16, 'Diamond 2': 17, 'Diamond 3': 18,
        'Ascendant 1': 19, 'Ascendant 2': 20, 'Ascendant 3': 21,
        'Immortal 1': 22, 'Immortal 2': 23, 'Immortal 3': 24,
        'Radiant': 25,
        'Placement': 8
    }
    df['rank_numeric'] = df['rank'].map(rank_mapping)
    validation_games = 30
    validation_data = df[-validation_games:]
    training_data = df[:-validation_games]
    data = training_data['rank_numeric'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    seq_length = 30
    X, y = create_sequences(data_scaled, seq_length)
    model = joblib.load('model.joblib')
    last_sequence = data_scaled[-seq_length:]
    predictions = []

    for _ in range(validation_games):
        current_seq = last_sequence.reshape(1, seq_length, 1)
        next_rank = model.predict(current_seq, verbose=0)
        predictions.append(next_rank[0])
        last_sequence = np.append(last_sequence[1:], next_rank)

    predictions = scaler.inverse_transform(np.array(predictions))
    agent_graph = get_top_agents_graph(df)
    map_graph = get_top_maps_graph(df)
    timeseries_graph = timeseries(df, validation_data, predictions)
    return agent_graph, map_graph, timeseries_graph

@app.route('/')
def index():
    df = pd.read_csv('valorant_games.csv')
    agent_graph, map_graph, timeseries_graph = process_data(df)
    return render_template('index.html', agent_graph=agent_graph, map_graph=map_graph, timeseries_graph=timeseries_graph)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        global user_df
        user_df = pd.read_csv(file)
        agent_graph, map_graph, timeseries_graph = process_data(user_df)
        return render_template('index.html', agent_graph=agent_graph, map_graph=map_graph, timeseries_graph=timeseries_graph)

if __name__ == '__main__':
    app.run(debug=True)