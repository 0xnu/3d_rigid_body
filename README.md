## Predicting 3D Rigid Body Dynamics with Deep Residual Network

3d_rigid_body is a hybrid C++ and Python project for simulating and predicting 3D rigid body dynamics. It combines a C++ physics engine for accurate simulation with a PyTorch-based deep residual network for learning and predicting complex 3D object interactions and movements.

### Default Execution

Execute the script with default parameters:

```sh
make run_python

## OR

make all
```

### Custom Execution

Execute the script with custom parameters:

```sh
make run_python_custom PYTHON_ARGS="[enter your custom arguments]"
```

#### Available Arguments

- `--input_size`: Size of the input tensor (13 for state + 6 for forces/torques). Default: 19
- `--output_size`: Size of the output tensor (13 for final state). Default: 13
- `--num_blocks`: Number of residual blocks in the network. Default: 10
- `--num_epochs`: Number of training epochs. Default: 200
- `--learning_rate`: Initial learning rate for the optimizer. Default: 0.001
- `--batch_size`: Batch size for training. Default: 64
- `--weight_decay`: Weight decay (L2 regularization) for the optimizer. Default: 0.0001
- `--ece_weight`: Weight for Energy Conservation Error in the loss function. Default: 0.1

#### Example Custom Execution

Here's an example of how to execute the script with custom parameters:

```sh
make run_python_custom PYTHON_ARGS="--input_size 32 --output_size 32 --num_blocks 15 --num_epochs 300 --learning_rate 0.0005 --batch_size 128 --weight_decay 0.00005 --ece_weight 0.2"
```

The command above will:

- Set the input and output sizes to 32
- Use 15 residual blocks
- Train for 300 epochs
- Use a learning rate of 0.0005
- Use a batch size of 128
- Apply weight decay of 0.00005
- Set the Energy Conservation Error weight to 0.2 in the loss function

You can adjust these parameters based on your specific requirements and the characteristics of your 3D rigid body dynamics prediction task.

### License

This project is licensed under the [GNU General Public License v3.0](./LICENSE).

### Citation

```tex
@misc{p3dodwdrn2024,
  author       = {Oketunji, A.F.},
  title        = {Predicting 3D Rigid Body Dynamics with Deep Residual Network},
  year         = 2024,
  version      = {0.0.1},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.12669002},
  url          = {https://doi.org/10.5281/zenodo.12669002}
}
```

### Copyright

(c) 2024 [Finbarrs Oketunji](https://finbarrs.eu). All Rights Reserved.