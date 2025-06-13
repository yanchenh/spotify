import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from utils import SpotifyDataset, VAE, train_vae

csv_path = "C:/Users/yanch/Desktop/project/SpotifyFeatures.csv"
dataset = SpotifyDataset(csv_path)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

input_dim = dataset[0].shape[0]
vae = VAE(input_dim=input_dim, latent_dim=10)
train_vae(vae, dataloader, epochs=50)

vae.eval()
latent_zs = []
raw_xs = []
with torch.no_grad():
    for x in dataloader:
        mu, _ = vae.encode(x)
        latent_zs.append(mu)
        raw_xs.append(x)
Z = torch.cat(latent_zs)
X_original = torch.cat(raw_xs)

target_vector = {
    'danceability': 0.9,
    'energy': 0.85,
    'valence': 0.8,
    'acousticness': 0.1,
    'instrumentalness': 0.2
}
df = pd.read_csv(csv_path).select_dtypes(include=[np.number]).dropna()
feature_names = df.columns.tolist()
target_vec_full = np.array([target_vector.get(f, 0.5) for f in feature_names])
scaler = dataset.scaler
target_scaled = scaler.transform(pd.DataFrame([target_vec_full], columns=feature_names))[0]

def decode_samples(model, z_samples):
    with torch.no_grad():
        return model.decode(z_samples).numpy()

def preference_score(sample, target):
    return -np.linalg.norm(sample - target)

z_samples = Z[:200]
x_samples = decode_samples(vae, z_samples)
y = np.array([preference_score(x, target_scaled) for x in x_samples])
train_x = z_samples.detach()
train_y = train_y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
z_min = train_x.min(0)[0]
z_max = train_x.max(0)[0]
train_x_unit = (train_x - z_min) / (z_max - z_min)

gp = SingleTaskGP(train_x_unit.double(), train_y.double())
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

ucb = UpperConfidenceBound(gp, beta=0.1)
bounds = torch.stack([torch.full((10,), -3.0), torch.full((10,), 3.0)])

bounds = torch.stack([torch.zeros_like(z_min), torch.ones_like(z_max)])
top_z = []
for _ in range(5):
    z_unit_opt, _ = optimize_acqf(
        acq_function=ucb,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=100,
    )
    top_z.append(z_unit_opt.squeeze(0))

z_nexts_unit = torch.stack(top_z)
z_nexts = z_nexts_unit * (z_max - z_min) + z_min

full_df = pd.read_csv(csv_path).dropna()
full_df = full_df[full_df["genre"].isin(["Pop", "Dance", "EDM", "Reggaeton"])]

feature_df = full_df.select_dtypes(include=[np.number])
meta_df = full_df.drop(columns=feature_df.columns)

scaled_real = scaler.transform(feature_df)

decoded_features = decode_samples(vae, z_nexts)
decoded_original = scaler.inverse_transform(decoded_features)

seen_indices = set()

for i, song_vec in enumerate(decoded_original):
    song_vec_df = pd.DataFrame([song_vec], columns=feature_names)
    dists = np.linalg.norm(scaled_real - scaler.transform(song_vec_df), axis=1)

    sorted_idx = np.argsort(dists)

    for idx in sorted_idx:
        if idx not in seen_indices:
            closest_idx = idx
            seen_indices.add(idx)
            break

    print(f"\nRecommended Track #{i+1}")
    print(meta_df.iloc[closest_idx].to_string(index=True))
    print("\nFeatures:")
    for k, v in zip(feature_names, feature_df.iloc[closest_idx]):
        print(f"{k:20s}: {v:.3f}")