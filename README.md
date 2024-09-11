# Vision Transformers for Weakly-Supervised enumeration of microorganisms

This repo allows to replicate experiments of the comparative study of different architecture approaches for the task of
weakly supervised enumeration.

## Description

This repository is prepared for training and testing different architectures like normal CNNs, REsNets, and Vision
Transformer approaches like the CrossViT, DeepViT, Parallel, XCiT and the vanilla ViT. The architectures are created for
the task of regression, which means that the output is a single number, designed so to they are able to count instances
instead of classifying.

## Datasets

There are four datasets used in this study:

* Fluorescent Neuronal Cells: Collection of mice neurons under fluroescence.
    * https://amsacta.unibo.it/id/eprint/6706/
* VGG cells: Collection of blue fluorescent cells.
    * https://www.robots.ox.ac.uk/~vgg/research/counting/index_org.html
* U2OS and HL-60 cancer cells: Collection of two cancer cell lines for counting task.
    * https://zenodo.org/records/4428844

### Special dataset created for this project:

* Artificial fluorescent bacteria dataset: Artificially created dataset that replicate the fluorescence of Bacillus
  Subtilis bacteria. The dataset used for the project is of large size, but access to it can be achieved by making use
  of the dataset generation tool developed for it:
  * https://github.com/JavierUrenaPhDProjects/artificial_fluorescent_dataset


## Getting Started

### Dependencies

The project runs on pytorch 3.10.

### Executing scripts

All arguments are configurable in the utils/config.py script, but can also be defined in the execution.

* Training the model
  ```
  python train.py --dataset yellow_cells --pretrain y --img_size 384 --model vit_base --epochs 100 --batch_size 32 --lr 0.0001
  ```
* Test architectures (WIP*)
  ```
  python test.py --dataset yellow_cells --img_size 384 --batch_size 16
  ```
  *Recommended using the class "TestAfterTrain" to create a tester after loading correctly the weights.

## Authors

Contributors names and contact info

ex. Javier Ureña Santiago  
ex. javier.urena@uibk.ac.at

## Acknowledgments

This project is part of the research and development project DesDet in collaboration with the department of Analytical
Chemistry and Radiochemistry, Hollu Systemhygiene GmbH and Planlicht GmbH \& Co KG. This project is funded by
Standortagentur Tirol.