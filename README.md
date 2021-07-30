# Desription

This is a template repository, which can be used to initialize new repositories for deep learning fast prototyping. The template allows training for single-node single-GPU models, single-node multi-GPU models, or multi-node multi-GPU models with SLURM. Checkpointing and auto-requeuing is supported on slurm as well.

# Usage

- Create a new repository on GitHub with name: [new-repo-name]
- Follow these steps:

```
$ git clone https://github.com/ramyamounir/Template.git
$ mv Template [new-repo-name]
$ cd [new-repo-name]
$ git remote set-url origin https://github.com/ramyamounir/[new-repo-name].git
$ git add .
$ git commit -m "Template copied"
$ git push
```

# Layout

This template follows a modular approach where main components of the code (archcitecture, loss, scheduler, trainer, etc.) are organized into subdirectories.

- The [train.py](train.py) script contrains all the arguments (parsed by argparse) and nodes/GPUS initializer (slurm or local). It also contains code for importing the dataset, model, loss function and passing them to the trainer function.
- The lib/trainer/trainer.py script defines the detailes of the training procedure.
- The lib/dataset/[args.dataset].py imports data and defines the dataset function. Creading a data directory with soft link to the dataset is recommended, especially for testing on multiple datasets.
- The lib/core/ directory contains definitions for loss, optimizer, scheduler functions.
- The lib/util/ directory contains helper functions organized by file name. (i.e. helper functions for distributed training are placed in the lib/util/distributed.py file).

# Run

## for single node, single GPU training:

Try the following example
```
python train.py -gpus 0 -model test
```

## for single node, multi GPU training:

Try the following example
```
python train.py -gpus 0,1,2 -model test
```

## for single node, multi GPU training on SLURM:

Try the following example
```
python train.py -slurm -slurm_nnodes 1 -slurm_ngpus 4 -slurm_partition general  -model test
```

## for multi node, multi GPU training on SLURM:

Try the following example
```
python train.py -slurm -slurm_nnodes 2 -slurm_ngpus 8  -slurm_partition general  -model test
```

### Tips

- To get more information about available arguments run: ```python train.py -h```
- To automatically start tensorboard server as a different thread, add the argument: ``` -tb ```
- To overwrite model log files and start from scratch add the argument: ``` -reset ```, otherwise it will use the last weights as checkpoint and add continue writing to the same tensorboard log files - if the same model name is used.
- To choose specific node names on SLURM, use the argument: ``` -slurm_nodelist GPU17,GPU18 ``` as an example. 
- If running on a GPU with Tensor cores, using mixed precision models can speed up your training. Add the argument ``` -fp16 ``` to try it out. If it makes training unstable due to the loss of precision, dont use it :)
- The template allows you to switch architectures, datasets and trainers easily by passing different arguments. For example, different architectures can be added to the lib/arch/[arch-name].py directory and passing the arguments as ``` -arch [arch-name] ``` or ``` -trainer [trainer-name] ``` or ``` -dataset [dataset-name] ```
- if you find an bug in this template, open a new issue, or a pull request. Any collaboration is more than welcome!

# License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.