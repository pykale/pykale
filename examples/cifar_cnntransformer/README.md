# Image Classification: Standard CNN vs. CNN+Attention

### 1. Description
In this example we train a standard 8 layer CNN on CIFAR10 as a baseline. We then take the same CNN architecture and stack a Transformer-Encoder ontop and train this new CNNTransformer model from scratch. We present several different variants of this model where we only alter the Transformer size. Below, the validation accuracy of each model is compared.

![Model Comparisons](CIFAR10-ModelComparison-ValAcc.png)
(The code example on this page does not aim to show any meaningful results, but only demonstrate the use of the CNNTransformer)

### 2. Observations
We observe that all models ultimately converge to a similar accuracy, only that the CNNTransformers learn slower which I think is likely because they use Dropout and the two base CNNs do not. 

### 3. Usage
`python main.py --cfg configs/one_of_the_config_files.yaml`
