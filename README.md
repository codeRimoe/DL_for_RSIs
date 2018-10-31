# DL_for_RSIs

##### Deep Leaning (DL) for remote sensing image (RSI) classification.

##### If it helps, please STAR the project and CITE our papers.

### Sample-set Maker(SSM) V5.0

- SSM.py

This is a class definition scipts for Sample-set Maker. SSM is a sample maker for RSI (Remote Sencing Image) classification, specifically for deep learning classification algorithm.
With SSM, you can easily load a RSI and get samples for training. Every Sample made by SSM is a N*N sub-image, which can be a sample of its center pixel for CNN/ResNet classifier.
Also, AL (Active Learning) is supported in SSM.

The script is compliant with PEP-8 specifications.

---

### Networks

#### 1. Wide Contextual Residual Network (WCRN)

This is a wide contextual residual network (WCRN) with active learning (AL) for remote sensing image (RSI) classification.

Though ResNets have achieved great success in various applications, its performance is limited by the requirement of abundant labeled samples. As it is very difficult and expensive to obtain class labels in real world, we integrate the proposed WCRN with AL to improve its generalization by using the most informative training samples.

Specifically, we first design a WCRN for RSI classification, and then integrate it with AL to achieve good machine generalization with limited number of training sampling. Experimental results on Pavia University and Flevoland datasets demonstrate that the proposed WCRN with AL can significantly reduce the needs of samples.

Environment:
> We run the scripts in Windows OS. </br>
> Spyder with Python 3.6 </br>
> Keras 2.0.8 using Tensorflow 1.2.1 backend </br>
> SSM.py </br>

Script:
> WCRN.py:         the definition of the network. </br>
> SSM.py:          a sample manager. </br>
> PU_train.py:     an example script for Pavia University, used for training. </br>
> PU_predict.py:   an example script for Pavia University, used for predicting.

The scripts are compliant with PEP-8 specifications.

Reference:

[Shengjie Liu, Haowen Luo, Ying Tu, Zhi He, and Jun Li. Wide Contextual Residual Network with Active Learning for Remote Sensing Image Classification. In International Geoscience and Remote Sensing Symposium, IGARSS 2018. (Accepted)](https://www.igarss2018.org/Papers/viewpapers.asp?papernum=2482)


#### 1. Shorten Spatial-spectral RNN with Parallel-GRU (St-SS-pGRU)

Convolutional neural networks (CNNs) attained a good performance in hyperspectral sensing image (HSI) classification, but CNNs consider spectra as orderless vectors. Therefore, considering the spectra as sequences, recurrent neural networks (RNNs) have been applied in HSI classification, for RNNs is skilled at dealing with sequential data. However, for a long-sequence task, RNNs is difficult for training and not as effective as we expected. Besides, spatial contextual features are not considered in RNNs. In this study, we propose a Shorten Spatial-spectral RNN with Parallel-GRU (St-SS-pGRU) for HSI classification. A shorten RNN is more efficient and easier for training than band-by-band RNN. By combining converlusion layer, the St-SSpGRU model considers not only spectral but also spatial feature, which results in a better performance. An architecture named parallel-GRU is also proposed and applied in St-SS-pGRU. With this architecture, the model gets a better performance and is more robust.

Environment:
> We run the scripts in Windows OS. </br>
> Jupyter with Python 3.6 </br>
> Tensorflow 1.8.0 </br>
> SSM.py

Script:
> Experiment.ipynb

Reference:

[Haowen Luo. Shorten Spatial-spectral RNN with Parallel-GRU for Hyperspectral Image Classification. arXiv preprint arXiv:1810.12563, 2018.](https://arxiv.org/abs/1810.12563)
