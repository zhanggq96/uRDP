# uRDP
Code for Universal Rate-Distortion-Perception Representations for Lossy Compression
<p float="center">
  <img src="https://i.imgur.com/DjCtjdM.png"/> 
</p>
A compression system with single encoder f and multiple decoders/discriminators. Top: model optimized for low distortion. At low rates, this causes blurriness. Bottom: high perceptual quality. Blurriness is eliminated, but the reconstruction is less faithful to original.

## Overview

This work follows up on the development of perceptually-enhanced lossy compression using machine learning done by Blau & Michaeli (2019) and Tschannen et al. (2018). These works have observed the somewhat counterintuitive phenomenon that when performing compression, optimizing for low distortion (e.g. MSE, SSIM) does not necessarily correlate with high perceptual quality as judged by a human observer. They re-introduce a rate-distortion-perception function from information theory literature, which generalizes the classical rate-distortion function with a distribution constraint acting as a mathematical proxy for perceptual quality. For a given rate, there is then a tradeoff between optimizing for low distortion and high perceptual quality; whereas the rate-distortion-perception function considers a separate encoder-decoder model for each tradeoff point, this work considers the possibility of reusing a single encoder to approximately achieve this tradeoff. 

## Requirements
See `requirements.txt`. It should work on most builds.

## Training

The first stage of training replicates the results of Blau & Michaeli (2019) by varying a parameter lambda to achieve various distortion-perception tradeoff points. We refer to these as end-to-end models. The distortion loss used is MSE and the perception loss is estimated through the discriminator of Wasserstein GAN. It is worth noting here that many common statistical measures can be used as a proxy for perceptual quality, through developments such as e.g. f-GAN. Finding the best such measure is an active field of research which is far more broad than the scope of this paper, and as such we choose Wasserstein GAN for simplicity and stability.

A base encoder from the first stage is then chosen to have its weights frozen and used to train new decoders/discriminators along different tradeoff points. These are compared to the tradeoff points from end-to-end models. The end result of our paper seems to indicate that there is very little to lose by reusing the encoder in this way on MNIST and SVHN.

<p float="center">
  <img src="https://i.imgur.com/vBjpOuR.png"/> 
</p>
Bolded points: end-to-end models (encoder and decoder/discriminator both trained). Unbolded points: universal models (frozen encoder, only decoder/discriminator trained). See `code_to_recreate_experiments.py` for the settings used to recreate the experiments in the paper.