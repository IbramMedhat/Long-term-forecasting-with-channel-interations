# Gated Time Series Forecasting with Channel Interactions

Main contributions in this repo include :
- Implementation of the TSMixer model done in [1] with experiments being run according to the best parameters configured in the paper.
- Implementation of a new time series forecasting model based on a gating mechanism.

## Baselines to compare with
- General Code structure, trasoformers experiments and Linear models experiments are taken from https://github.com/cure-lab/LTSF-Linear which is implementation for the paper "Are Transformers Efficient for Time Series Forecasting?(AAAI 2023) [2]
- results are compared to transformers [3,4,5,6,7], Linear models [2] and TSMixers [1].

## Detailed Description
We provide all experiment script files in `./scripts`:
| Files      |                              Interpretation                          |
| ------------- | -------------------------------------------------------| 
| EXP-LongForecasting      | Long-term Time Series Forecasting Task                    |
| EXP-LookBackWindow      | Study the impact of different look-back window sizes   | 
| EXP-Embedding        | Study the effects of different embedding strategies      |

Other github repos which the code is based on :

The implementation of Autoformer, Informer, Transformer is from https://github.com/thuml/Autoformer

The implementation of FEDformer is from https://github.com/MAZiqing/FEDformer

The implementation of Pyraformer is from https://github.com/alipay/Pyraformer

## Gating-based Time Series Forecasting Model


### Comparison with Transformers
Univariate Forecasting:
![image](pics/Uni-results.png)
Multivariate Forecasting:
![image](pics/Mul-results.png)
LTSF-Linear outperforms all transformer-based methods by a large margin.

### Efficiency
![image](pics/efficiency.png)
Comparison of method efficiency with Look-back window size 96 and Forecasting steps 720 on Electricity. MACs are the number of multiply-accumulate operations. We use DLinear for comparison, since it has the double cost in LTSF-Linear. The inference time averages 5 runs.

## Getting Started
### Environment Requirements

First, please make sure you have installed Conda. Then, our environment can be installed by:
```
conda create -n LTSF_Linear python=3.6.9
conda activate LTSF_Linear
pip install -r requirements.txt
```

### Data Preparation

You can obtain all the four benchmarks from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. All the datasets are well pre-processed and can be used easily.

```
mkdir dataset
```
**Please put them in the `./dataset` directory**

### Training Example
- In `scripts/ `, we provide the model implementation *Dlinear/Autoformer/Informer/Transformer*
- In `FEDformer/scripts/`, we provide the *FEDformer* implementation
- In `Pyraformer/scripts/`, we provide the *Pyraformer* implementation


![image](pics/Visualization_DLinear.png)

# References

[1]
[2]
[3]
[4]
[5]
[6]
[7]
[8]