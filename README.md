<h2 style="font-size: 1.5em" align="center">
  Learning Nonlinear Dynamics in Physical Modelling Synthesis using Neural Ordinary Differential Equations
</h2>

<p style="font-size: 1.0em" align="center">
  Victor Zheleznov, Stefan Bilbao, Alec Wright and Simon King
</p>

<p style="font-size: 1.0em" align="center">
  Accompanying repository for the DAFx25 paper
</p>

<div align="center">
  
  [![arXiv](https://img.shields.io/badge/arXiv-2505.10511-b31b1b.svg)](https://arxiv.org/abs/2505.10511)
  [![Sound Examples](https://img.shields.io/badge/Sound_Examples-blue)](https://victorzheleznov.github.io/dafx25/)
  
</div>

## Repository Contents

`audio/` includes all sound examples for the datasets used in the paper. Some of these sound examples are presented on the accompanying web-page.

`cfg/` includes configuration files for experiments.

`src/` includes source code for datasets, generators, models, losses and other utils.

`out/` includes configurations and checkpoints for trained models.

`generate.py` is a script for dataset generation.

`train.py` is a script for training the model.

`test.py` is a script for testing the trained model.



## Instructions

### Environment Setup

[Python 3.11.9](https://www.python.org/downloads/release/python-3119/) was used for simulations.
The required packages are proved in the `requirements.txt` file. To setup the environment, use:
```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data Generation

The `cfg/dataset` folder includes configuration files for dataset generation listed in the table below.
Please refer to the paper for more details on chosen dataset parameters.

| File | Description |
| -------- | ------- |
| `nonlinear_oscillator_dataset_cubic.yaml`   | Nonlinear oscillator dataset with cubic nonlinearity |
| `nonlinear_oscillator_dataset_sinh.yaml`    | Nonlinear oscillator dataset with hyperbolic sine nonlinearity |
| `nonlinear_string_dataset_train_valid.yaml` | Nonlinear string dataset used for training and validation |
| `nonlinear_string_dataset_test.yaml`        | Nonlinear string dataset used for testing |

To generate data, use:
```
python -m generate --config-name=nonlinear_oscillator_dataset_cubic
```
The generated dataset will be saved within the `data` folder.

### Training

The `cfg` folder includes `nonlinear_oscillator` and `nonlinear_string` configuration files which match experimental
setup described in the paper. To train the model, use:
```
python -m train --config-name=nonlinear_oscillator
```

If you have a [WandB](https://wandb.ai/) account, you can enable online logging:
```
python -m train --config-name=nonlinear_oscillator writer.mode=online writer.project=project_name
```

The training run will be saved within the `out` folder with a time stamp, e.g., `out/yyyy_mm_dd_HH-MM-SS_nonlinear_oscillator`.
Checkpoint with the lowest validation loss `checkpoint_best.pt` and
checkpoint from the last epoch `checkpoint_*.pt` will be saved.

### Testing

To test the trained model, you need to specify the output folder as a configuration path and a dataset partition:
```
python -m test ---config-path=out/yyyy_mm_dd_HH-MM-SS_nonlinear_oscillator +partition=valid
```

If `--config-path` is not specified the script will use the last output folder. Computed model predictions
will be stored within the `test` folder.

### Results Reproduction

To reproduce simulation results from the paper, you need to use *exactly* the same datasets.
At the moment, these can be obtained by a personal request to [v.zheleznov@ed.ac.uk](mailto:v.zheleznov@ed.ac.uk?subject=Datasets%20Request). After the downloaded datasets are placed within the `data` folder, use the following command to train the model for nonlinear string:
```
python -m train --config-name=nonlinear_string
```
You can compare the trained model with checkpoints provided in  the `out/2025_03_20_19-20-07` folder — they should be identical.

To test the trained model on the test dataset, use the following configuration overrides:
```
python -m test \
"dataset.fs=96000" \
"dataset.dur=3" \
"dataset.gamma_range=[130, 246]" \
"dataset.kappa_range=[1.01, 1.1]" \
"dataset.sigma0_range=[2, 2]" \
"split=[0, 0, 1]" \
"+partition=test"
```

The source code includes an option to use JIT compilation for numerical solver (the `USE_JIT` flag within the `src/models/modal_system.py` file). This flag should be set to `False` (which is a default option) as a compiled solver can produce slightly different numerical results.



## Citation

If you use this work, please use the following citation:
```
@inproceedings{Zheleznov2025,
    author = {V. Zheleznov and S. Bilbao and A. Wright and S. King},
    title = {{L}earning {N}onlinear {D}ynamics in {P}hysical {M}odelling {S}ynthesis using {N}eural {O}rdinary {D}ifferential {E}quations},
    booktitle = {Proc. 28th Int. Conf. Digital Audio Effects},
    year = {2025}
}
```