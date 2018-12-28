## Trust Region Adversarial Attack

TRAttack is a pytorch library for Trust Region Based Adversarial Attack on neural networks that could be used in conjunction with pytorch. The 
Library currently supports utility functions to compute adversarial examples of different neural network models.

## Example Usage
After installing pytorch, and training a neural network, the following command generates adversarial examples and computes the relative perturbations:

python trattack_cifar.py --resume cifar10_result/model_params.pkl --test-batch-size 1000 --worst-case 0 --iter 2000 --norm 8 --eps 0.001 --adap

For ImageNet, we can use the pretrained model from pytorch:

python trattack_imagenet.py -a resnet50 --pretrained --batch-size 1 --worst-case 0 --iter 5000 --norm 8 --eps 0.0001 --class 9 --plotting image_example/ --adap

## Citation
TRAttack has been developed as part of the following paper. We appreciate ig if you would please cite the following if you 
found the library useful for your work:

* Z Yao, A Gholami, P Xu, K Keutzer, MW Mahoney. Trust Region Based Adversarial Attack on Neural Networks, [PDF](https://arxiv.org/pdf/1812.06371.pdf)
