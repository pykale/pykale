# bandgap-benchmark

Code for “Benchmarking Band Gap Prediction for Semiconductor Materials Using Multimodal and Multi-fidelity Data”.

This repository contains the PyTorch Lightning implementation of the benchmark described in our paper, *Benchmarking Band Gap Prediction for Semiconductor Materials Using Multimodal and Multi-fidelity Data*. We compile a new multimodal, multi-fidelity dataset from the Materials Project and BandgapDatabase1, consisting of **60,218** low-fidelity (computational) band gaps and **1,183** high-fidelity (experimental) band gaps. We evaluate seven ML models, including three traditional methods (linear regression, random forest regression, and support vector regression) and four GNNs (CGCNN, CartNet, LEFTNet-Z, and LEFTNet-Prop).

## Data

Expected layout (adjust paths in your config YAML):

- Place a JSON file in `data/` that lists Materials Project IDs (mpids) and target values.
- Place the corresponding CIF files in `cif_file/`, named as `<mpid>.cif`.

Example:
```(bash)
data/
fine_tune/
train_data.json
val_data.json
test_data.json
cif_file/
mp-123456.cif
mp-654321.cif
```

## Config

To run `main.py`, create a YAML file (passed via `--cfg`) based on `config.py`. Key sections:

- **DATASET**: Paths to train/val/test JSON files.
- **MODEL**:
  - `NAME`: choose one of `cgcnn`, `leftnet`, `cartnet`, `random_forest`, `linear_regression`, `svr`.
  - `CIF_FOLDER`: directory containing all crystal `.cif` files.
  - `INIT_FILE`: JSON with atomic feature encodings.
  - `PRETRAINED_MODEL_PATH`: checkpoint to load (optional).
  - `MAX_NBRS`, `RADIUS`: neighbor-search settings used in preprocessing.
- **SOLVER**: training settings — `EPOCHS`, `LR`, `BATCH_SIZE`, `OPTIM` (`SGD`/`Adam`), `SEED`, `WORKERS`.
- **CGCNN / LEFTNET / CARTNET**: model-specific options.
  - For **LEFTNet**, set `ENCODING` to `"z"` (one-hot atomic number) or `"prop"` (precomputed atomic properties). See paper for details.
- **LOGGING**: `LOG_DIR`, `PROJECT_NAME` — where checkpoints/logs are saved.


## Training

To train a model (add `--pretrain` to perform pretraining once instead of k-fold training):

```(bash)
python main.py --cfg configs/PATH_TO_YOUR_CONFIG.yaml
# pretrain only:
# python main.py --cfg configs/PATH_TO_YOUR_CONFIG.yaml --pretrain
```
After training, predictions can be generated using:

```(bash)
python test_model.py \
  --cfg configs/PATH_TO_YOUR_CONFIG.yaml \
  --checkpoint saved_models/PATH_TO_YOUR_MODEL.ckpt \
  --cif_folder cif_file \
  --test_data data/fine_tune/test_data.json
```

## Related `kale` API

- `kale.loaddata.materials_datasets`: Load CIF files and extract features for crystal structures.
- `kale.embed.gcn`: Graph-convolutional building blocks and encoders for graph data.
- `kale.embed.materials_equivariant`: SE(3)-equivariant components for materials GNNs—RBF distance encoders, smooth cutoffs, invariant neighbor initialization, local-frame message passing, and related utilities.
- `kale.pipeline.base_nn_trainer`: Regression trainer used in the benchmark.
- `kale.evaluate.metrics`: Metrics/utilities used in the benchmark (e.g., Gaussian distance for encoding interatomic distances; mean relative error for evaluation).
