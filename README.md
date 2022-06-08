## A ResNet Benchmark on CIFAR-100 and TinyImageNet for Neural Network Verification

We propose a new set of benchmarks on CIFAR-100 and TinyImageNet datasets with ResNet-based model architectures.

**Model details**: We provide four well-trained ResNet models on CIFAR-100, with flexible model sizes, and one ResNet model for TinyImageNet. These models are trained with [CROWN-IBP](https://github.com/huanzhang12/CROWN-IBP) and adversarial training. The smallest model is 11-layer and the largest model has 21 layers. The models include:

* CIFAR-100
  * resnet-small: 4 residual blocks, 9 convolutional layers + 2 linear layers. 
  * resnet-medium: 8 residual blocks, 17 convolutional layers + 2 linear layers.
  * resnet-large: Almost identical to standard resnet18 architecture (8 residual blocks, 17 convolutional layers), but with 2 linear layers.
  * resnet-super: 9 residual blocks, 19 convolutional layers + 2 linear layers.
* TinyImageNet:
  * resnet-medium: 8 residual blocks, 17 convolutional layers + 2 linear layers.

The ONNX format networks are available in the [onnx](https://github.com/Lucas110550/CIFAR100_TinyImageNet_ResNet/tree/main/onnx) folder.

The detailed info of each model can be founded in this table:

|           Model            | Model size | Number of layers | Clean acc. | CROWN verified acc. $\epsilon=1/255$ | PGD acc. $\epsilon=1/255$ |
| :------------------------: | :--------: | :--------------: | :--------: | :----------------------------------: | :-----------------------: |
|   CIFAR100_resnet_small    |    5.4M    |      11          |   51.61%   |                20.14%                |          37.92%           |
|   CIFAR100_resnet_medium   |   10.1M    |      19          |   54.57%   |                29.08%                |          40.42%           |
|   CIFAR100_resnet_large    |   15.2M    |      19          |   53.24%   |                29.89%                |          39.31%           |
|   CIFAR100_resnet_super    |   31.6M    |      21          |   53.95%   |                27.53%                |          38.84%           |
| TinyImageNet_resnet_medium |   14.4M    |      19          |   35.04%   |                13.51%                |          21.85%           |

**Verification Properties**: We use standard robustness verification properties with perturbation ε=1/255.

**Data Format**: The input image should be normalized by using the mean and std computed from CIFAR-100 and TinyImageNet dataset. Specifically, we provide the normalizer we used in our evaluation here:

* CIFAR-100: mean (0.5071, 0.4865, 0.4409), std (0.2673, 0.2564, 0.2761)
* TinyImageNet: mean (0.4802, 0.4481, 0.3975), std (0.2302, 0.2265, 0.2262)

**Data Selection**: We randomly select 10 images from the test set for CIFAR100_resnet_small, 16 for CIFAR100_resnet_super and 24 for other models. These images are classified correctly and cannot be attacked by a 100-steps PGD attack with 20 random restarts. We also filtered out the samples which can be verified by vanilla CROWN (which is used during training) to make the benchmark more challenging. The filtering process is done offline on a machine with a GPU due to the large sizes of these models.

**Generating properties**: Run the following command to generate vnnlib files and the csv file with a random seed:

```
python generate_properties.py <seed>
```
