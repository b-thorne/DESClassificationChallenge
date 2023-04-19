# fairuniverse 

## Cosmology Challenge Description

### Overview

In order to discover new examples of interesting time-varying astrophysical sources, such as AGN or supernovae, cosmologists must sift through an extensive list of viable "candidates" identified in the large volume of collected survey observations. This volume of data is already too large for human experts to address, and is directly addressable by machine learning methods.

In this challenge, users will use data from the Dark Energy Survey (DES) to design a machine learning model for the automatic detection of transient astrophysical sources.

### Data 

The data in this challenge was first collated for classification tasks for use in the DES supernova transient detection pipeline in [Goldstein et al 2015](https://arxiv.org/abs/1504.02936), and was subsequently updated in [Ayyar et al (2022)](https://arxiv.org/abs/2203.09908). Each sample in the dataset consists of three images centered on a candidate source, using data from DES operations in 2013 and 2014. 

This source may either be a simulated astrophysical transient of scientific interest that has been injected, or a real artifcat produced by the pipeline (for example by imperfect beam modeling, imprecise astrometry leading to off-centered differencing). 

There are 898,693 independent samples, each consisting three 51x51 pixel images ("search", "template", and "difference"), and have an associated label of either 1 (artifact) or 0 (non-artifact). The total dataset size is 56 GB. Below is a plot of three independent data points, one on each row, with each image type in a different column. 

![](example_plot.png)

### Evaluation

This is a binary classification problem. Each triplet of images can be characterized as an artifcat (true) or a non-artifact (false). The classifier is scored by the AUC metric. 

![](fpt_mdr.png)

## Challenge?

For context, the dataset is about 56GB, and the simple CNN model used can be trained on a single A100 in less than an hour. This makes it a reasonable scale for a first challenge. 

### Area of improvement

- The "real" point sources in this dataset were generated by injecting simulated sources into difference maps. When the trained model was used for inference for real DECam data, it had a missed detection rate of ~50%. Using an additional expert-labelled dataset to improve the base model, Graham et al 2023 were able to reduce the missed detection rate to ~5%. 
- The requirement of expert labellers means that we are likely to miss the rare, brighter events, which will not be well-represented in the training data, and may be missed by the base simulation-trained algorithm. It was shown in Ayyar et al 2022, that even higher S/N images from the raw DECam dataset were missed by the simulation-trained algorithm. 
- Systematically missing out rare, brighter, sources can lead to biases in the cosmology constraints derived from these supernova detections. 
- How can we frame a challenge to address this shortcoming? 

### Other topics 

- Generative modeling problems, e.g. current work by Peter / Zarija.


## To-Do 

- Pytorch data loader tuning 
- Mmap 
- Bias disable for pre batch norm layers 
- Difference between gif and fits?
- Make volumes nice 
- Calculate normalizations over batches 
- Do some data augmentation
- Do  
