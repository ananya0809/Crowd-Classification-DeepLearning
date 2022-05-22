# Classification of Crowd Density into Sparse, Medium and Dense using Deep Learning
## Research Oriented Minor Project

### Problem Statement:
A classification model that can classify a given image of a crowd into Sparse or Medium or Dense based on crowd density using Deep Leanring methodologies.

### Reference Links: 

- [Dataset](https://www.kaggle.com/datasets/tthien/shanghaitech)
- [About Dataset](https://paperswithcode.com/dataset/shanghaitech)
- [Reference Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)
- [MCNN Implementation Code Reference](https://github.com/svishwa/crowdcount-mcnn)
- [HeatMap Generation](https://github.com/oldj/pyheatmap)

### Installation & Setup
1. Clone this repository using the below command:

```git clone git@github.com:ananya0809/Crowd-Classification-DeepLearning.git```

This will setup a remote working directory to store files and dataset in.

2. Download the [dataset](https://www.kaggle.com/datasets/tthien/shanghaitech) available for ShanghaiTech.
4. [Import libraries](https://github.com/oldj/pyheatmap) to draw a heatmap from a random dummy data of any crowd dataset.
5. Reset path variables code block as mentioned in ```final.ipynb``` to the path variable on your system as per requirement.

### Data Preparation
1. The folder ```data_preparation``` has 2 main ```MATLAB``` scripts which parses the input images and generates patches of image for training and test data.
2. Edit the ```create_gt_test_set_shtech.m``` and ```create_training_set_shtech.m``` accordingly as per the path variables in your system.
3. Run both the scripts so as to formulate the model pre-training.

### Idea
The idea is to implement classification of crowd images into 3 classes namely 'Sparse', 'Medium', & 'Dense' based on a ```Decision Tree``` logic where target variable is set manually for training whereas, for testing and validation the model generates a ```heatmap``` from a random input image using the trained ```MCNN``` model that will aid in classification from scratch without any prior ```ground truth``` data available for the random input image, and later classify based on the trained decision tree available.

Idea Representation:

![Idea]("https://github.com/ananya0809/Crowd-Classification-DeepLearning/blob/main/readme_images/Frame%2028.png")

### Training
##### Part 1:
1. In reference to the code in ```final.ipynb``` jupyter notebook, an end-to-end demonstration of density heatmap generation using the reference code of 'HeatMap Generation' is done.
2. It allows the image from the crowd dataset to input into an MCNN [Multi Column Neural Network] architecture where there are 3 sequential CNN models joined together to create a system of deep neural nets that enable feature recognition and extraction at each sequential model from face, torso and overall body respectively.

Model Architecture Representation:

![MCNN Architecture]("https://github.com/ananya0809/Crowd-Classification-DeepLearning/blob/main/readme_images/MCNN.png")

3. The generation of heatmap is obtained from the reference code using ```pyHeatMap``` and stored in a sub-directory.

Image Before Heatmap Generation

![Pre-HeatMap]("https://github.com/ananya0809/Crowd-Classification-DeepLearning/blob/main/readme_images/heat_B_2_pre_51.png")

Actual Image After Heatmap Generation

![Act-HeatMap]("https://github.com/ananya0809/Crowd-Classification-DeepLearning/blob/main/readme_images/heat_B_2_act_51.png")

##### Part 2:
1. Now that the dataset has been modified from simple crowd images to heatmap generated images, the (x,y) coordinates of each image which are pre-present in a ```.mat``` file available in the ```data_subset``` folder are taken into account to implement the Decision Tree Classifier.
2. The ```ground-truth``` data available in the ```.mat``` file is used in implementation of K-means clustering where K is taken as 5 for all images.

K-Means Clustering with K=3 (an example)

![Pre-Cluster]("https://github.com/ananya0809/Crowd-Classification-DeepLearning/blob/main/readme_images/kmeans0.png")

![Cluster-Plot]("https://github.com/ananya0809/Crowd-Classification-DeepLearning/blob/main/readme_images/kmeans1.png")

![Image-Clustered]("https://github.com/ananya0809/Crowd-Classification-DeepLearning/blob/main/readme_images/kmeans2.png")

3. On the obtained clusters a connectivity model approach is implemented where each datapoint containing (x,y) coordinates from 1st cluster is measured in distance from every other datapoint in the 2nd cluster. Thus for K=5, 10 pairs of clusters are obtained where the distance between every datapoint is calculated using ```Euclidean Distance```.
4. From the obtained distances, the minimum distance is taken and stored in a ```.csv``` file, that is generated to map minimum distances per cluster pair to its respective image.
5. This ```.csv``` is then converted into a ```Pandas DataFrame``` to define columns and also add another column for ```Target``` class.
6. The target classes are filled manually for the training phase so that a decision tree can be generated for testing and validation.

NOTE: K-Means is an unsupervised machine learning technique that is unable to generate labels on its own during clustering, thus it is necessary to manually input labels for classification by training the model on respective classes.

##### Part 3:
1. Now the generated DataFrame is finally used for training and classification of the labelled images into their respective classes.
2. The source columns are the minimum distance calculated per cluster pair for a given image and the target variable is the class for which the image has to be classified.
3. For classification, a Decision Tree is generated based on the ```CART``` algorithm where the GINI Index is calculated for all attributes that are the 10 cluster pairs as per the input target value.
4. A confusion matrix is obtained for the same that reports the ```accuracy``` as 83.33% for the limited sub-dataset provided to train the model after manual classification.
5. A report containing ```precision```, ```recall```, ```f1-score```, ```support``` is also calculated for the same.

Output containing Confusion Matrix, Accuracy and the Report for trained model:

![Accuracy-Train]("https://github.com/ananya0809/Crowd-Classification-DeepLearning/blob/main/readme_images/Screenshot%202022-05-20%20at%2010.11.56%20AM.png")

### Testing and Validation
##### Part 4:
1. This part of the code is the implementation for generation of heatmaps using ```pyHeatMap``` library as referenced from the given link.
2. The model for the same along with the learning weights are taken from a ```model.json``` file.
3. The path variable to the input image which is not used for training is updated.
4. A heatmap for the given input image is generated. This model also predicts the count of people recognized by the model from the provided input image.

NOTE: Here the heatmap obtained consists of multiple datapoints containing respective (x,y) coordinates separated at a pixel's distance from each other over each feature vector recognized. When multiple features (nearly 1000) are recognized by the model the intensity of the heatmap changes from blue colour to red color at that particular location.

Before Processing

![Pre-Heatmap]("https://github.com/ananya0809/Crowd-Classification-DeepLearning/blob/main/readme_images/output.png")

After Processing

![Post-Heatmap]("https://github.com/ananya0809/Crowd-Classification-DeepLearning/blob/main/readme_images/output1.png")

##### Part 5:
1. This part is purely testing the model.
2. Here, we generate the combination of all distances for all cluster pairs and append the minimum most distance into the generated ```.csv``` file.
3. Lastly, prediction using Decision Tree Classifier is obtained through the help of ```GINI Index```.
4. The label for the input image is generated with the help of this Decision Tree and thus, the model helps in 'Classification of Crowd' into 'Sparse', 'Medium' or 'Dense' labels depending upon its density.
### Collaborated by:
- [Ananya Agrawal](https://github.com/ananya0809)
- [Hardik Srivastava](https://github.com/oddlyspaced)

