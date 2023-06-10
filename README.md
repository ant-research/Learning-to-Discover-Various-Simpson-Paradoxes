# Learning to Discover Various Simpson’s Paradoxes

This repository includes the implementation details of the method ```SimNet``` and helps readers to reproduce the results in the paper **Learning to Discover Various Simpson’s Paradoxes**（KDD 2023）.

SimNet is a simple neural network model with high operational efficiency.


## Requirements
- Python 3.6+
- PyTorch 1.13.0
- pandas	1.5.1
- scipy	1.9.3
- numpy	1.23.4
- statsmodels	0.13.5

## Datasets
The repository includes:
- Altanla city employee salary (CES) data
- Auto mile per gallon (MPG) data
- UC Berkeley admission data
- Titanic data
- Synthetic data of different noise levels mentioned in our paper.

## Methods
The repository includes three methods to discover Simpson's paradox. Methods can be chosen when one creates an object of SimpsonParadoxFinder.
- ```naive```: An implementation of the simple method from paper "*Detecting Simpson’s Paradox*" (FLAIRS 2018).
- ```trend```: An implementation of the ```Trend Simpson's Paradox Algorithm``` from paper "*Can you Trust the Trend?: Discovering Simpson's Paradoxes in Social Data*" (WSDM 2018).
- ```simnet```: The one proposed in our paper (KDD 2023).

## Hyper-parameter Usage of SimNet
- ```hidden_dim```: the dimension of hidden layers, set large (e.g. 256) when the number of features is large.
- ```learning_rate```: set small (e.g. 0.001) when the number of samples is large.
- ```find_strong_amp```: set ```True``` if expect to discover 'AMP2' when $PCC_{T,Y}>0$ or to discover 'AMP1' when $PCC_{T,Y}<0$.

## Quick Start
An example for Simpson's paradox discovery using SimNet on Iris data.

```python
python run_iris.py
```

## Citation
If you find this project useful in your research, please consider cite our paper:

*Jingwei Wang, Jianshan He, Weidi Xu, Ruopeng Li, and Wei Chu. 2023. Learning to Discover Various Simpson’s Paradoxes. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’23), August 6–10, 2023, Long Beach, CA, USA. ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/3580305.3599859*

## Acknowledgment
We thank Dr. Yu Wu (who has left Ant Group) for his excellent technical support for this project.