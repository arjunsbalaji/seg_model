# IVOCT Segmentation

This project is an effort to produce robust, timely and accurate 3D geometries of coronary arteries from \
Intravascular OCT imaging. 

In this notebook presentation I will show you how to train, validate and test a [Capsule Network](https://towardsdatascience.com/capsule-networks-the-new-deep-learning-network-bd917e6818e8) on a dataset of coronary artery IVOCT B-scans. This work was done at VascLabs and was designed with current fluid solver software in mind. 

Some of the key goals and aims of this project:
1. production of a deep learning model that is at least state of the art at binary IVOCT segmentation, measured by the [Dice Score](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient).
2. the final masks should be compatible with modern fluid solver software; be reliable and produce smooth (or at least easily smoothable) masks.
3. segment whole B-scans in under 30 seconds. This is required for feasible use in theatre.

In this talk I will show you the data we are working with, how we build the model, how we train and validate the model and some analysis of the results at the end. 

I will explain concepts as we go along, but if you have any questions please ask away! 
![Image] ('./nbs/useful/a10worstbestmiddle.jpg')

## This is currently a work in progress and is under review for publication

**Things to note**
- './nbs/Analysis.ipynb' is a good place to start to see the notebook workflow and experimentation.
- './src/' contains the original implementation of the capsule segmentation network.
- We use MLflow to log experiments 
- Build Dockerfile to get a ready working environment. Personally I find Docker really great because it allows for a production ready environment that we can easily use to push this to a cloud service provider (AWS/Azure/GCS), or supercomputing service like Pawsey.


