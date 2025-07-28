# Progressive Poster Dataset Distillation
PyTorch Implementation for the "Progressive Poster Dataset Distillation" paper.  

___

> **A Well Distilled Poster is Worth a Thousand Images**<br>
> Spyridon Giakoumatos, Nema Arpita, Lin Weisi<br>
> *<br>
>
>**Abstract:** Dataset distillation is a process that aims to create a small set of synthetic images that enables models trained on it to achieve
performance comparable to training on the full
dataset. Current state-of-the-art algorithms like
Poster Dataset Distillation (PoDD) train models on
a highly compressed distilled dataset, traditionally
refining a single synthetic poster iteratively.
However, this approach discards initial versions of
posters that contain synthesized images rich in low-
level information, thereby limiting knowledge
retention and model generalization. In this paper,
we extend PoDD to a multi-stage learning
framework, where multiple evolving versions of the
dataset are stored and progressively reused during
training. Rather than relying exclusively on the
latest distilled dataset, our method incorporates
prior versions to preserve earlier learned
information, mitigating catastrophic forgetting and
improving model stability. A key insight is that a key
metric in dataset distillation is not just the quality of
the final dataset but the cumulative knowledge
retained across training stages. We introduce
Progressive Poster Dataset Distillation (P-PoDD),
which dynamically integrates information from all
previous dataset stages using novel blending and
refinement techniques to improve knowledge
retention. Experiments on CIFAR-10 demonstrate
that P-PoDD improves test accuracy by 4.46% over
PoDD at 0.9 images per class, while maintaining
similar computational overhead. By leveraging
multiple evolving dataset versions, P-PoDD
achieves better generalization, improved
robustness, and enhanced learning efficiency—
establishing a new paradigm for dataset distillation
grounded in progressive knowledge accumulation.




## Installation 
1.  Clone the repo:
```bash
git clone https://github.com/AsafShul/PoDD
cd PoDD
```
2. Create a new environment with needed libraries from the `environment.yml` file, then activate it:
```bash
conda env create -f environment.yml
conda activate podd
```

## Dataset Preparation
This implementation supports the following 4 datasets:
- [CIFAR-10](https://paperswithcode.com/dataset/cifar-10)
- [CIFAR-100](https://paperswithcode.com/dataset/cifar-100)


#### CIFAR-10 and CIFAR-100
Both the CIFAR-10 and CIFAR-100 datasets are built-in and will be downloaded automatically. 

## Running PoDD
The `main.py` script is the main script in this project.

#### CIFAR-10
```bash
python main.py --name=PPoDD-CIFAR10-LT1-90 --distill_batch_size=96 --patch_num_x=16 --patch_num_y=6 --dataset=cifar10 --num_train_eval=8 --update_steps=1 --batch_size=5000 --ddtype=curriculum --cctype=2 --epoch=10000 --test_freq=10 --print_freq=10 --arch=convnet --window=60 --minwindow=0 --totwindow=200 --inner_optim=Adam --outer_optim=Adam --inner_lr=0.001 --lr=0.001 --syn_strategy=flip_rotate --real_strategy=flip_rotate --seed=0 --zca --comp_ipc=1 --class_area_width=32 --class_area_height=32 --poster_width=153 --poster_height=60 --poster_class_num_x=5 --poster_class_num_y=2 --num_stages=5
```

#### CIFAR-100
```bash
python main.py --name=PoDD-CIFAR100-LT1-90 --distill_batch_size=50 --patch_num_x=20 --patch_num_y=20 --dataset=cifar100 --num_train_eval=8 --update_steps=1 --batch_size=2000 --ddtype=curriculum --cctype=2 --epoch=10000 --test_freq=10 --print_freq=10 --arch=convnet --window=100 --minwindow=0 --totwindow=300 --inner_optim=Adam --outer_optim=Adam --inner_lr=0.001 --lr=0.001 --syn_strategy=flip_rotate --real_strategy=flip_rotate --seed=0 --zca --comp_ipc=1 --class_area_width=32 --class_area_height=32 --poster_width=303 --poster_height=303 --poster_class_num_x=10 --poster_class_num_y=10 --train_y --num_stages=5
```


#### Important Hyper-parameters
- `--patch_num_x` and `--patch_num_y` - The number of extracted overlapping patches in the x and y axis of the poster.
- `--poster_width` and `--poster_height` - The width and height of the poster (controls the distillation data budget).
- `--poster_class_num_x` and `--poster_class_num_y` - The class layout dimensions within the poster as a 2d array (e.g., 10X10 or 20X5), 
(the product must be equal to the number of classes).
- `--train_y` - If set, the model will also optimize a set of learnable labels for the poster.
- `--num_stages` - Determines how many PPoDD stages will be executed, defaults to 5 if not defined.


## Acknowledgments
- This repo uses [*PoDD*](https://arxiv.org/pdf/2403.12040) as the underlying Poster algorithm, code found in their supplementary materials.
