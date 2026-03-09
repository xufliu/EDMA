**EDMA** is an energy-based formulation to improve the model's explanation fidelity for 3D molecular graph models—currently **SchNet** and **DimeNet++**.

## Setup Environment

This is an example for how to set up a working conda environment to run the code.

```shell
conda create -n edma python=3.9
conda activate edma

```

### Install PyTorch + PyG

> Torch/PyG wheels are platform-specific. Install them **before** the rest.

## Quickstart (Instance-level Explanation)

Run an end-to-end explanation on **QM9**:

```bash
python energy_explainer_instance_qm9.py
```



## License

Released under the **MIT License**. See [LICENSE](LICENSE).