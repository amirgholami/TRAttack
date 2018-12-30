## Trust Region Adversarial Attack

TRAttack is a pytorch library for Trust Region Based Adversarial Attack on neural networks. The 
library currently supports utility functions to compute adversarial examples for different neural network models.

## Example Usage
First execute the following commands:

git clone git@github.com:amirgholami/TRAttack.git
git submodule update --init

After training a neural network, the following command generates adversarial examples and computes the relative adversarial perturbation norm:

python trattack_cifar.py --resume cifar10_result/model_params.pkl --test-batch-size 1000 --worst-case 0 --iter 2000 --norm 8 --eps 0.001 --adap

For ImageNet, one can also use a pretrained model from pytorch as follows:

python trattack_imagenet.py -a resnet50 --pretrained --batch-size 1 --worst-case 0 --iter 5000 --norm 8 --eps 0.0001 --class 9 --plotting image_example/ --adap

## Citation
TRAttack has been developed as part of the following paper.  If you found the library useful for your work, we appreciate if you would please cite the following paper:

* Z Yao, A Gholami, P Xu, K Keutzer, MW Mahoney. Trust Region Based Adversarial Attack on Neural Networks, [PDF](https://arxiv.org/pdf/1812.06371.pdf)
