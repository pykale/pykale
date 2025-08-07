### bandgap-benchmark
Code for "Benchmarking Band Gap Prediction For Semiconductor Materials Using Multimodal And Multi-fidelity Data"

This repository contains the PyTorch Lightning implementation of the benchmark that described in our paper "Benchmarking Band Gap Prediction For Semiconductor Materials Using Multimodal And Multi-fidelity Data". We compiled a new multimodal, multi-idelity dataset from the Materials Project and BandgapDatabase1, consisting of 60,218 low-fidelity computational band gaps and 1,183 high-fidelity experimental band gaps. We evaluated seven ML models, including three traditioanl methods (linear regression, random forest regression and support vector regression) and four GNN (CGCNN, CartNet, LEFTNet-Z and LEFTNet-Prop). 

### Repository Structure
`cif_file.zip` - Contains `.cif` files and the atomic encoding file used in the benchmark.

`data/` - Directory containing MPIDs and corresponding band gap values:
* `pretrain_data.json` - 60,218 PBE band gap values.
* `fine_tune/` - Experimental band gap values.
* `data_by_type/` - Data used for "leave-one-material-out" splits, categorized by material type.
`configs/` - Configuration files for training models.

`models/` - Implementations of baseline models.

`loaddata/` - Data preparation, splitting, and processing.

`leave_one_material_out/` - Scripts and data for running leave-one-material-out experiments.

`saved_models` - Pretrained models.

### Installation

Install dependencies with:

```(bash)
pip install -r requirements.txt
```

### Training

To train a model, use the following command (add `--pretrain` to perform pretraining only once instead of k-fold training):

```(bash)
python main.py --cfg configs/PATH_TO_YOUR_CONFIG.yaml
```
After training, predictions can be generated using:

```(bash)
python test_model.py --cfg configs/PATH_TO_YOUR_CONFIG.yaml --checkpoint saved_models/PATH_TO_YOUR_MODEL.ckpt --cif_folder cif_file --test_data data/fine_tune/test_data.json
```

