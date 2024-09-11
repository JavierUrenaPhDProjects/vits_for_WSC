# Vision Transformers for Weakly-Supervised enumeration of microorganisms

This repo allows to replicate experiments of the comparative study of different architecture approaches for the task of 
weakly supervised enumeration. 

## Description

This repository 

## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* something
* something

### Executing scripts
Here i Show some example scripts executions. All arguments are configurable in the utils/config.py script, but can also be defined in the execution. 

* Training the model
  ```
  python train.py --dataset cell_counting_yellow --pretrain y --img_size 384 --model TC_384_16_gap --epochs 500 --batch_size 16 --lr 0.00001
  ```
* Test architectures
  ```
  python test.py --dataset cell_counting_yellow --img_size 384 --batch_size 16
  ```
* Generate predictions in images
  ```
  python test_images.py --dataset cell_counting_yellow --img_size 384 --batch_size 16
  ```


## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Javier Ureña Santiago  
ex. javier.urena@uibk.ac.at

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release. Created utils.py, model.py and dataloader.py

## License



## Acknowledgments
