# DL_for_RSIs

### Sample-set Maker(SSM) V5.0

- SSM.py

This is a class definition scipts for Sample-set Maker. SSM is a sample maker for RSI (Remote Sencing Image) classification, specifically for deep learning classification algorithm.
With SSM, you can easily load a RSI and get samples for training. Every Sample made by SSM is a N*N sub-image, which can be a sample of its center pixel for CNN/ResNet classifier.
Also, AL (Active Learning) is supported in SSM.

### Networks

####1. Wide Contextual Residual Network (WCRN)
This is a wide contextual residual network (WCRN) with active learning (AL) for remote sensing image (RSI) classification.

Though ResNets have achieved great success in various applications, its performance is limited by the requirement of abundant labeled samples. As it is very difficult and expensive to obtain class labels in real world, we integrate the proposed WCRN with AL to improve its generalization by using the most informative training samples.

Specifically, we first design a WCRN for RSI classification, and then integrate it with AL to achieve good machine generalization with limited number of training sampling. Experimental results on Pavia University and Flevoland datasets demonstrate that the proposed WCRN with AL can significantly reduce the needs of samples.

Reference:
Shengjie Liu, Haowen Luo, Ying Tu, Zhi He, and Jun Li. Wide Contextual Residual Network with Active Learning for Remote Sensing Image Classification. In International Geoscience and Remote Sensing Symposium, IGARSS 2018. (Accepted)
https://www.igarss2018.org/Papers/viewpapers.asp?papernum=2482