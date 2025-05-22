# Trained Models

Trained models used in the paper are listed in the table below.

| Folder | Description |
| -------- | ------- |
| `2025_03_20_19-20-07` | Trained nonlinear string model |
| `2025_03_30_13-17-45` | Trained nonlinear oscillator model with cubic nonlinearity |
| `2025_03_30_13-18-31` | Trained nonlinear oscillator model with hyperbolic sine nonlinearity |

Each folder contains:
- `cfg.yaml` file with training run configuration;
- `checkpoint_best.pt` checkpoint with the lowest validation loss;
- `checkpoint_5000.pt` checkpoint from the last epoch.
