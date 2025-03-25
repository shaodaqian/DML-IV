

# DML-IV

This repository includes the implementation of the method proposed in "Learning Decision Policies with Instrumental Variables through Double Machine Learning", in proceedings of ICML 2024. [Paper Link.](https://arxiv.org/abs/2405.08498)(https://proceedings.mlr.press/v235/shao24d.html)

## Citation

```
@inproceedings{shao2024dmliv,
    title={Learning Decision Policies with Instrumental Variables through Double Machine Learning},
    author={Daqian Shao, Ashkan Soleymani, Francesco Quinzan and Marta Kwiatkowska},
    year={2024},
    booktitle={Proceedings of the 41rd International Conference on Machine Learning},
}
```

## Dependencies

Install all dependencies

```
pip install -r requirements.txt
```



## Basic Usage

Run experiments using DML-IV, CE-DML-IV using the aeroplane demand dataset with low-dim, high-dim and the two real world datasets.

```
python main_low_d.py
python main_mnist.py
python main_realworld.py
```
