import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from mamba_ssm import Mamba
import pandas as pd
import os
import time

np.random.seed(42)
torch.manual_seed(42)


# Parameters

features = ['temp', 'humidity', 'wind_speed', 'wind_dir', 'rainfall', 'pressure']
target = 'pm25'
SEQ_LEN = 168  # 7 days
HORIZON = 24   # next 24h
hidden_dim = 128


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


# Sequence creation

def create_sequences(city_df, seq_len=SEQ_LEN, horizon=HORIZON):
    X = city_df[features].values
    y = city_df[target].values.reshape(-1,1)
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len - horizon + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len:i+seq_len+horizon].flatten())
    return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)


# Pure Mamba Forecast Model

class MambaForecast(nn.Module):
    def __init__(self, in_features, hidden_dim, horizon):
        super(MambaForecast, self).__init__()
        self.mamba = Mamba(
            d_model=in_features,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        # x: (B, T, F)
        x_out = self.mamba(x)       # (B, T, F)
        x_pool = x_out.mean(dim=1)  # (B, F)
        x = F.relu(self.fc1(x_pool))
        out = self.fc2(x)
        return out

# Prepare DataLoaders with 80/10/10 split

all_X, all_y, all_city = [], [], []
cities = df_scaled['city_x'].unique()

for city in cities:
    city_df = df_scaled[df_scaled['city_x']==city].reset_index(drop=True)
    X_seq, y_seq = create_sequences(city_df)
    all_X.append(X_seq)
    all_y.append(y_seq)
    all_city.extend([city]*len(X_seq))

all_X = torch.cat(all_X, dim=0)
all_y = torch.cat(all_y, dim=0)
all_city = np.array(all_city)

N = len(all_X)
train_end = int(0.8 * N)
val_end   = int(0.9 * N)

train_dataset = TensorDataset(all_X[:train_end], all_y[:train_end])
val_dataset   = TensorDataset(all_X[train_end:val_end], all_y[train_end:val_end])
test_dataset  = TensorDataset(all_X[val_end:], all_y[val_end:])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Train the model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MambaForecast(len(features), hidden_dim=hidden_dim, horizon=HORIZON).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 10
for epoch in range(epochs):
    # ---- Training ----
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    train_loss = total_loss / len(train_loader.dataset)

    # Validation 
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")


# Evaluate on test set
model.eval()
mamba_results = {}

y_min, y_max = df_final[target].min(), df_final[target].max()

with torch.no_grad():
    for city in cities:
        city_df = df_scaled[df_scaled['city_x']==city].reset_index(drop=True)
        X_seq, y_seq = create_sequences(city_df)
        loader = DataLoader(TensorDataset(X_seq, y_seq), batch_size=64, shuffle=False)

        y_true_list, y_pred_list = [], []
        start_time = time.time()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            y_pred_list.append(out.cpu().numpy())
            y_true_list.append(yb.cpu().numpy())

        inference_time = time.time() - start_time

        y_pred = scaler_y.inverse_transform(np.vstack(y_pred_list))
        y_true = scaler_y.inverse_transform(np.vstack(y_true_list))

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        r, _ = pearsonr(y_true.flatten(), y_pred.flatten())
        nrmse = rmse / (y_max - y_min)

        mamba_results[city] = {
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


# Metrics table

results_table = []
for city, metrics in mamba_results.items():
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

results_df = pd.DataFrame(results_table)
out_dir = "mamba_without_aod"
os.makedirs(out_dir, exist_ok=True)
print("Evaluation Metrics  (MAMBA) without AOD Data:\n")
results_df.to_csv(os.path.join(out_dir, "mamba_metrics.csv"), index=False)



# Plot results

for city in cities:
    city_df = df_final[df_final['city_x']==city].reset_index(drop=True)
    y_true = mamba_results[city]['y_true']
    y_pred = mamba_results[city]['y_pred']

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
    plt.title(f"MAMBA Forecast without AOD Data - {city}")
    plt.xlabel("Datetime")
    plt.ylabel("PM2.5")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plot_path = os.path.join(out_dir, f"{city}_forecast.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {plot_path}")


# Compute NMSE per horizon step

horizon_nmse = {city: [] for city in cities}

with torch.no_grad():
    for city in cities:
        city_df = df_scaled[df_scaled['city_x'] == city].reset_index(drop=True)
        X_seq, y_seq = create_sequences(city_df)
        loader = DataLoader(TensorDataset(X_seq, y_seq), batch_size=64, shuffle=False)

        y_true_list, y_pred_list = [], []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            y_pred_list.append(out.cpu().numpy())
            y_true_list.append(yb.cpu().numpy())

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
plt.title("NMSE per Forecast Horizon (1–24h) across all Cities (MAMBA)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
nmse_plot_path = os.path.join(out_dir, "nmse_per_horizon.png")
plt.savefig(nmse_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {nmse_plot_path}")
#plt.show()



