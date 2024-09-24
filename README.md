
The dataset of AQI-36 is from Yi et al.[1], which has already stored in ./data/pm25/. The datasets of PEMS-BAY are from Li et al.[2],

[1] X. Yi, Y. Zheng, J. Zhang, and T. Li, “St-mvl: filling missing values in geo-sensory time series data,” in Proceedings of the 25th International Joint Conference on Artificial Intelligence, 2016


### training and imputation for the healthcare dataset ```shell python exe_physio.py --testmissingratio [missing ratio] --nsample [number of samples]


## Acknowledgements  A part of the codes is based on CSDI https://github.com/ermongroup/CSDI
@inproceedings{tashiro2021csdi,  
title={CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation},   
author={Tashiro, Yusuke and Song, Jiaming and Song, Yang and Ermon, Stefano}, 
booktitle={Advances in Neural Information Processing Systems},   
year={2021} }
