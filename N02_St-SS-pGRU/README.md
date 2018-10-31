## Wide Contextual Residual Network (WCRN)

Author  : Shengjie Liu, Haowen Luo (Equal Contribution)

License : http://www.apache.org/licenses/LICENSE-2.0

If it helps, please STAR the project and CITE our papers.

Convolutional neural networks (CNNs) attained a good performance in hyperspectral sensing image (HSI) classification, but CNNs consider spectra as orderless vectors. Therefore, considering the spectra as sequences, recurrent neural networks (RNNs) have been applied in HSI classification, for RNNs is skilled at dealing with sequential data. However, for a long-sequence task, RNNs is difficult for training and not as effective as we expected. Besides, spatial contextual features are not considered in RNNs. In this study, we propose a Shorten Spatial-spectral RNN with Parallel-GRU (St-SS-pGRU) for HSI classification. A shorten RNN is more efficient and easier for training than band-by-band RNN. By combining converlusion layer, the St-SSpGRU model considers not only spectral but also spatial feature, which results in a better performance. An architecture named parallel-GRU is also proposed and applied in St-SS-pGRU. With this architecture, the model gets a better performance and is more robust.

Environment:
> We run the scripts in Windows OS. </br>
> Jupyter with Python 3.6 </br>
> Tensorflow 1.8.0 </br>
> SSM.py

Script:
> Experiment.ipynb

The Hyperspetral RSI of Pavia University is available at: [Hyperspectral Remote Sensing Scenes - Grupo de Inteligencia Computacional (GIC)](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)


Reference:

[Haowen Luo. Shorten Spatial-spectral RNN with Parallel-GRU for Hyperspectral Image Classification. arXiv preprint arXiv:1810.12563, 2018.](https://arxiv.org/abs/1810.12563)
