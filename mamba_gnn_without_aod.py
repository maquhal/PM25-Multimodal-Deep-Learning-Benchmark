import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import torch.nn as nn
from mamba_ssm import Mamba
import os
import time

np.random.seed(42)
torch.manual_seed(42)


# Parameters

features = ['temp', 'humidity', 'wind_speed', 'wind_dir', 'rainfall', 'pressure']
target = 'pm25'
SEQ_LEN = 168  # 7 days
HORIZON = 24   # next 24h
hidden_dim = 128  # GNN hidden features


# Load dataset

df_final = pd.read_csv("dataset.csv")


# Scale features and target

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_all = df_final[features].values
y_all = df_final[target].values.reshape(-1,1)

X_all = np.nan_to_num(X_all, 0.0)
y_all = np.nan_to_num(y_all, 0.0)

X_scaled = scaler_x.fit_transform(X_all)
y_scaled = scaler_y.fit_transform(y_all)

df_scaled = df_final.copy()
for i, f in enumerate(features):
    df_scaled[f] = X_scaled[:, i]
df_scaled['pm25'] = y_scaled


# Create graph sequences

def create_graph_sequence(city_df, seq_len=SEQ_LEN, horizon=HORIZON):
    X = city_df[features].values
    y = city_df[target].values.reshape(-1,1)
    graphs = []
    for i in range(len(X) - seq_len - horizon + 1):
        x_seq = torch.tensor(X[i:i+seq_len], dtype=torch.float)
        y_seq = torch.tensor(y[i+seq_len:i+seq_len+horizon], dtype=torch.float).T  # shape (1, horizon)

        # Fully connected edges
        row, col = np.meshgrid(np.arange(seq_len), np.arange(seq_len))
        edge_index = torch.tensor(np.vstack([row.flatten(), col.flatten()]), dtype=torch.long)

        graphs.append(Data(x=x_seq, edge_index=edge_index, y=y_seq))
    return graphs


# Mamba block

def get_mamba_block(hidden_dim, seq_len):
    print("Using official mamba_ssm.Mamba")
    return Mamba(
        d_model=hidden_dim,
        d_state=16,
        d_conv=4,
        expand=2,
    )


# GNN + Mamba model

class GNN_Mamba_Forecast(nn.Module):
    def __init__(self, in_features, hidden_dim, horizon, seq_len):
        super(GNN_Mamba_Forecast, self).__init__()
        self.seq_len = seq_len
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.mamba = get_mamba_block(hidden_dim, seq_len)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, horizon)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        batch_size = int(batch.max().item()) + 1
        total_nodes = x.shape[0]
        expected = batch_size * self.seq_len

        if total_nodes != expected:
            print(f"Warning: total_nodes ({total_nodes}) != batch_size*seq_len ({expected}). Skipping Mamba.")
            x_mamba = x
        else:
            x_seq = x.view(batch_size, self.seq_len, -1)  # (B, T, D)
            x_seq_out = self.mamba(x_seq)  # (B, T, D)
            x_mamba = x_seq_out.contiguous().view(total_nodes, -1)

        x_pooled = global_mean_pool(x_mamba, batch)
        x = F.relu(self.fc1(x_pooled))
        out = self.fc2(x)
        return out


# Prepare DataLoaders

all_graphs = []
cities = df_scaled['city_x'].unique()
for city in cities:
    city_df = df_scaled[df_scaled['city_x']==city].reset_index(drop=True)
    city_graphs = create_graph_sequence(city_df)
    all_graphs.extend(city_graphs)

split = int(0.8 * len(all_graphs))
train_loader = DataLoader(all_graphs[:split], batch_size=64, shuffle=True)
test_loader  = DataLoader(all_graphs[split:], batch_size=64, shuffle=False)


# Train the model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN_Mamba_Forecast(len(features), hidden_dim=hidden_dim, horizon=HORIZON, seq_len=SEQ_LEN).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader.dataset):.6f}")


# Evaluate

model.eval()
gnn_results = {}

y_min, y_max = df_final[target].min(), df_final[target].max()

with torch.no_grad():
    for city in cities:
        city_df = df_scaled[df_scaled['city_x']==city].reset_index(drop=True)
        city_graphs = create_graph_sequence(city_df)
        loader = DataLoader(city_graphs, batch_size=64, shuffle=False)

        y_true_list, y_pred_list = [], []
        
        start_time = time.time()   # Start inference timer
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            y_pred_list.append(out.cpu().numpy())
            y_true_list.append(batch.y.cpu().numpy())
        end_time = time.time()     # End inference timer

        inference_time = end_time - start_time

        y_pred = scaler_y.inverse_transform(np.vstack(y_pred_list))
        y_true = scaler_y.inverse_transform(np.vstack(y_true_list))

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        r, _ = pearsonr(y_true.flatten(), y_pred.flatten())
        nrmse = rmse / (y_max - y_min)

        gnn_results[city] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'MSE': mse,
            'RMSE': rmse,
            'NRMSE': nrmse,
            'MAE': mae,
            'R2': r2,
            'R': r,
            'InferenceTimeSec': inference_time
        }

# Store metrics in DataFrame

results_table = []
for city, metrics in gnn_results.items():
    results_table.append({
        "City": city,
        "MSE": metrics['MSE'],
        "RMSE": metrics['RMSE'],
        "NRMSE": metrics['NRMSE'],
        "MAE": metrics['MAE'],
        "R2": metrics['R2'],
        "R": metrics['R'],
        "InferenceTimeSec": metrics['InferenceTimeSec']
    })


out_dir = "mamba_gnn_without_aod"
os.makedirs(out_dir, exist_ok=True)

results_df = pd.DataFrame(results_table)

results_csv_path = os.path.join(out_dir, "mamba_gnn_metrics.csv")
results_df.round(4).to_csv(results_csv_path, index=False)
print(f"Metrics saved to {results_csv_path}")



# Plot results

for city in cities:
    city_df = df_final[df_final['city_x']==city].reset_index(drop=True)
    y_true = gnn_results[city]['y_true']
    y_pred = gnn_results[city]['y_pred']

    y_true_plot = y_true[:, 0]
    y_pred_plot = y_pred[:, 0]

    datetimes = pd.to_datetime(city_df['datetime'].iloc[SEQ_LEN:SEQ_LEN+len(y_true_plot)])
    mask = (datetimes >= "2025-04-01") & (datetimes < "2025-05-01")
    datetimes_april = datetimes[mask]
    y_true_april = y_true_plot[mask]
    y_pred_april = y_pred_plot[mask]

    plt.figure(figsize=(12,5))
    plt.plot(datetimes_april, y_true_april, label='True', color='blue')
    plt.plot(datetimes_april, y_pred_april, label='Pred', color='orange')
    plt.fill_between(datetimes_april, y_pred_april-5, y_pred_april+5, color='purple', alpha=0.1)
    plt.title(f"GNN MAMBA Forecast without AOD - {city}")
    plt.xlabel("Datetime")
    plt.ylabel("PM2.5")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"{city}_forecast.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {plot_path}")

    #plt.show()


# Compute NMSE per horizon step

horizon_nmse = {city: [] for city in cities}

with torch.no_grad():
    for city in cities:
        city_df = df_scaled[df_scaled['city_x'] == city].reset_index(drop=True)
        city_graphs = create_graph_sequence(city_df)
        loader = DataLoader(city_graphs, batch_size=64, shuffle=False)

        y_true_list, y_pred_list = [], []
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            y_pred_list.append(out.cpu().numpy())
            y_true_list.append(batch.y.cpu().numpy())

        y_pred = scaler_y.inverse_transform(np.vstack(y_pred_list))
        y_true = scaler_y.inverse_transform(np.vstack(y_true_list))

        # NMSE for each horizon step (1..24)
        for h in range(HORIZON):
            mse_h = mean_squared_error(y_true[:, h], y_pred[:, h])
            var_h = np.var(y_true[:, h])
            nmse_h = mse_h / var_h if var_h > 0 else np.nan
            horizon_nmse[city].append(nmse_h)


# Plot NMSE per horizon step

plt.figure(figsize=(12,6))
for city in cities:
    plt.plot(range(1, HORIZON+1), horizon_nmse[city], marker='o', label=city)

plt.xlabel("Forecast Horizon (hours ahead)")
plt.ylabel("NMSE")
plt.title("NMSE per Forecast Horizon (1–24h) across all Cities (GNN + MAMBA without AOD)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

nmse_plot_path = os.path.join(out_dir, "nmse_per_horizon.png")
plt.savefig(nmse_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {nmse_plot_path}")


