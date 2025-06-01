# Personal Project for Practice: Human Protein Atlas

This project is based on a past [Kaggle Competition](https://www.kaggle.com/competitions/human-protein-atlas-image-classification/overview).

# Project Description and Goal
This project involves the use of machine learning to analyze four channel microscope imagery to determine the location of proteins in cells. The Human Protein Atlas seeks to better map out and understand human proteins. Looking at where proteins are present in cells gives important basic indications for the function of the protein, since each part of the cell has very specific functions. To do this, the Human Protein Atlas uses a fluorescent label that binds to the protein of interest on cell cultures. Then, by shining a light, microscope images are taken of the cells with highlighted fluorescent areas indicating the prescence of the protein of interest. Three other chemical labels are used in similar manners to highlight the nucleus, endoplasmic reticulum, and microtubules on the same cells. 

Production of these images are relatively easy. It is the analysis that takes expertise and time. As such, with many unlabled images, the Human Protein Atlas seeks to use machine learning to aid them. The goal of this competition was to use machine learning to analyze these images and be able to accurately identify in which part of the cell the proteins are present in, using the three other microscope images of other parts of the cells as useful references. 

Complexities of this project included: multi-class multi-label classification; extremely skewed label frequency distribution; four channel data; significant differences between images (multiple different types of cells); computing power limitations.

# Why I Chose this Project
This project was particularly interesting to me as I only had basic experience with simple machine learning models up to this point. I had never tackled a larger or more impactful project, and this project stood out as a balance of complexity and intrigue. I have always been interested in both computer science and biology. I hope to be able to apply my computer science and machine learning skills in truly rewarding and impactful areas of biology, such as the analysis of human protein functions and more. As such, I was drawn to this project as it mixed those two fascinating fields. 

This project was also not extraordinarely complicated, but not simple either by any means. It had its own complexities and issues that beginner courses had not taught me. For instance, each microscope image was seperated into four channels (protein, nucleus, endoplastic reticulum, and microtubles), which was somewhat unusual. Furthermore, the project involved multi-class multi-label classification with an extremely skewed label frequency distribution, which better reflects how AIs are often applied in real complex scenarios. As such, the complexities of this project helped me build a stronger basis in machine learning and allowed me to get important experience in applying machine learning to more complex problems and the various technical challenges and aspects at play. 

# Skills and Experiences Learned
- From this project, I was able to get more familiar with Python and PyTorch specifically, as well as associated libraries (NumPy, MatPlotLib-Py, and Pandas).
- I was also able to get more experience with handling datasets through Pandas and dealing with them in tensors through PyTorch. Data analysis is very important to real world projects where data is often not clean.
- I was also able to build a stronger understanding of CNNs, as well as some special CNN architectures such as the ResNet and DenseNet (and how information carried forward can improve accuracy).
- I also learned more about augmentation techniques to deal with images that are often very different from each other.
- I was also able to learn how to deal with multi-class multi-label problems with skewed data through the use of different loss functions and evaluation metrics (F1 score), regularization, optimizers, and augmentations to deal with this. 

# Approach Summary
1. ResNet50 pre-trained model with fine-tuning -> had to be adapted to take four channel inputs by changing layers and averaging starting weights. 50 epochs of training. 512x512 images used. 85/15 train/validation split. 
2. AdamW optimizer with weight decay and OneCycleLearningRate scheduler for faster convergence and noise reduction. 
3. Focal Cross Entropy loss (custum implementation) used to deal with the skewedness of the label frequency distribution.
4. Basic augmentations used on training data included random flips, 90 degree rotations, minor contrast and color adjustments, and normalization by mean and SD. Some augmentations required custom implementation for four channel data. 
6. Post-training analysis of model prediction distributions to further adjust true/false label thresholds.
7. Macro F1 Score to account for infrequent labels and skewedness.

0.71 Validation Score | 0.4 LB Score

# Possible Improvements
The major discrepancy between my validation and submission (LB) score is a problem that remains unsolved. It is likely due to some significant differences between the test and training data. Other participants also reported similar drops, though most people were able to achieve scores of around 0.44 which I was not able to reach. From learning curve analysis, it is clear that overfitting is a significant issue here. Nevertheless, that does not directly explain such a major discrepancy between validation and final test scores as both of those should be affected by overfitting to similar levels. 
Nevertheless, implementation of L1 or L2 regularization, as well as more substantial data augmentations may be beneficial to reduce overfitting and increase overall accuracy. 
Further changes to the data may also be beneficial, such as the removal of a 4th channel (microtubles) because of redundant information/similar imagery between microtubles and ER. 
Furthermore, the use of a stronger pre-trained backbone such as a DenseNet or EfficientNet (B3) has generally been reported to perform better than a ResNet, though the computationl costs are also substantial. 



