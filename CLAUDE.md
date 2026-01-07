# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PI-GANO (Physics-Informed Geometry-Aware Neural Operator) is a research implementation of a neural operator for solving partial differential equations (PDEs) on variable domain geometries. The model is geometry-aware and can be trained without Finite Element Method (FEM) data.

**Paper**: [Physics-Informed Geometry-Aware Neural Operator](https://www.sciencedirect.com/science/article/pii/S0045782524007941) (Computer Methods in Applied Mechanics and Engineering, 2025)

**Supported PDE Problems**:
- Darcy flow problem (groundwater flow)
- 2D plate stress problem (structural mechanics)

**Supported Models**:
- `GANO` - Geometry-Aware Neural Operator (proposed method)
- `DCON` - Deep Compositional Operator Network (baseline)
- `self_defined` - Custom model template for development

## Common Commands

```bash
# Install dependencies (Python 3.8+)
pip install -r requirements.txt

# Train GANO model
python PINO_darcy_training.py --model='GANO' --phase='train'
python PINO_plate_training.py --model='GANO' --phase='train'

# Evaluate GANO model
python PINO_darcy_training.py --model='GANO' --phase='test'
python PINO_plate_training.py --model='GANO' --phase='test'

# Train baseline DCON model
python PINO_darcy_training.py --model='DCON' --phase='train'
python PINO_plate_training.py --model='DCON' --phase='train'

# Train custom model
python PINO_darcy_training.py --model='self_defined' --phase='train'
```

**Command-line arguments**:
- `--model`: Model type ('GANO', 'DCON', 'self_defined')
- `--phase`: 'train' or 'test'
- `--geo_node`: Geometry node selection ('vary_bound' or 'all_domain')
- `--data`: Dataset name (default: 'Darcy_DG' or 'plate_stress_DG')

## Architecture Overview

### Entry Points
- `PINO_darcy_training.py` - Main script for Darcy flow problem
- `PINO_plate_training.py` - Main script for plate stress problem
- `NO_darcy_training.py` - Neural Operator baseline for Darcy
- `NO_plate_training.py` - Neural Operator baseline for plate stress

### Core Modules (`lib/`)
- `model_darcy.py` - Darcy problem models (PI_DCON, PI_GANO classes)
- `model_plate.py` - Plate stress problem models
- `utils_darcy_train.py` - Training loop and utilities for Darcy
- `utils_plate_train.py` - Training loop and utilities for plate stress
- `utils_data.py` - Data loading and preprocessing (handles variable-sized geometries via padding)
- `utils_losses.py` - Physics-informed loss functions for different PDEs

### Configuration (`configs/`)
Model hyperparameters are stored in YAML files named `{model}_{problem}.yaml`:
- `GANO_Darcy_DG.yaml` - GANO config for Darcy problem
- `GANO_plate_stress_DG.yaml` - GANO config for plate stress
- `DCON_Darcy_DG.yaml` - DCON config for Darcy
- `DCON_plate_stress_DG.yaml` - DCON config for plate stress

### Data Format
Expected `.mat` files contain:
- `u_field` - Solution values at collocation points
- `coors` - Coordinates of collocation points
- `BC_input_var` - Boundary condition parameters
- `totoal_ic_flag` - Flags indicating PDE vs boundary nodes

Place datasets in `data/` directory (e.g., `Darcy_DG.mat`, `plate_stress_DG.mat`).

### Model Architecture Pattern

**GANO** (`PI_GANO` class):
- Uses a geometry encoder (`DG` class) to process domain boundary information
- Combines parameter encoding with coordinate-based neural networks
- Variants: `PI_GANO`, `PI_GANO_add`, `PI_GANO_mul`, `PI_GANO_geo`

**DCON** (`PI_DCON` class):
- Physics-informed mesh-independent baseline
- Branch-trunk architecture typical of neural operators
- Branch network processes parameter inputs
- Trunk network processes coordinates

### Custom Model Development

To add a custom model architecture:

1. In `lib/model_darcy.py` or `lib/model_plate.py`, define your model class inheriting from `nn.Module` with `__init__(self, config)` and `forward()` methods
2. Create config files: `configs/self_defined_Darcy_star.yaml` and `configs/self_defined_plate_stress_DG.yaml`
3. Run with `--model='self_defined'`

The training utilities in `utils_darcy_train.py` and `utils_plate_train.py` can also be modified for custom training algorithms.
