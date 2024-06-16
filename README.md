# Information Theory Analysis of Supervised Learning using MNIST dataset

## Table of contents
* [Abstract](#abstract)
* [Introduction](#introduction)
* [Implementation of the Project](#implementation-of-the-project)
* [Output](#output)
  
## Abstract
This project, titled "Information Theory Analysis of Supervised Learning using MNIST dataset", aims to analyze supervised learning using information theory tools such as entropy and mutual information. The primary motivation is to understand the prediction power of each pixel in the MNIST dataset, a large database of handwritten digits commonly used for training various image processing systems.

The methodology involves importing and binarizing the MNIST dataset, plotting heatmaps of the digits, and calculating the mutual information between each pixel and its class label. The project uses Python and libraries such as numpy, scipy, sklearn, matplotlib, and seaborn.

The results include a heatmap representing the optimal shape of each digit and an analysis of the prediction ability of the pixels with the highest mutual information. The project demonstrates that information theory can provide valuable insights into supervised learning.

In conclusion, this project provides a novel approach to analyzing supervised learning using information theory. The results have potential implications for improving the effectiveness of image processing systems.

## Introduction
Computer vision is a specialised area within artificial intelligence (AI) that empowers computers to interpret and comprehend visual data from the real world. It is designed to mimic the human visual system’s capability to perceive and process visual data, such as images and videos. On the other hand, machine learning is a specific branch of artificial intelligence that concentrates on constructing systems capable of learning from data. These systems can make predictions or decisions without the need for explicit programming. This enables computers to identify patterns and make sense of intricate data. Computer vision and Machine learning work together to enable computers to understand and interpret visual information, opening up a wide range of applications across various industries. Supervised learning is a type of machine learning where it learns from labelled data, where the model is trained on input-output pairs. 

In this project, we are going to do supervised learning analysis on a popular dataset known as MNIST. The MNIST dataset is a collection of handwritten digits from 0 to 9 (around 70,000 images). Each digit is represented as a grayscale image of size 28x28 pixels. This means that each image is composed of 28 rows and 28 columns of pixels, and each pixel has a value representing the intensity of the grayscale (ranging from 0 which represents black to 255 which represents white). Our main tools for this analysis will be concepts from information theory, namely entropy and mutual information. 

<p align="center">
  <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/f86544b1-4e7c-4d17-b1a4-2704c9c226d8" alt="Description of the image" width="400"/>
  <br>
  <em>Several samples of “handwritten digit image” and its “label” from MNIST dataset</em>
</p>

Our first step will be to convert the MNIST dataset into a binary format. This process is known as binarization. Once we have our binarized data, we’ll create a visual representation of it. We will do this by adding up each digit’s values in the training dataset and displaying the result as a heatmap. This will give us a clear picture of the data we’re working with. Next, we will calculate mutual information. This will be done for each pixel and its corresponding class label. Mutual information is a measure of the relationship between two variables, in this case, a pixel and its class label. Finally, we will identify the pixels that have the highest mutual information. These pixels are important because they have a strong relationship with their class labels. We will then use these pixels to predict the class labels in the dataset. This will give us an idea of how well these pixels can predict the class labels.

<p align="center">
  <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/45becd30-1e1b-4951-b8a2-28714af53d8b" alt="Description of the image" width="400"/>
  <br>
  <em>Block Diagram of the Project Workflow</em>
</p>

## Implementation of the Project
The primary resource of the program, the MNIST dataset is imported using the fetch__openml function from scikit-learn. This function fetches datasets from the OpenML repository which is a collaborative platform for machine learning. We specifically import ‘mnist__784’ which refers to the MNIST dataset with 784 pixels (28x28). The MNIST dataset has a data attribute that stores the intensity of a pixel in an image (where 0 represents black and 255 represents white). This data is then binarized converting each pixel value in the images to either 0 or 1 based on a certain threshold or criterion.

In supervised learning we learn from labeled data where the model is trained on input-output pairs. The target attribute of the MNIST dataset is a variable that corresponds to each image, storing the digit present in that image. Although it can be stored as a floating point, we convert it to pure integers here.

To form training and testing sets, we split the binarized MNIST dataset and its labels into a testing set and a training set each of it contains 10,000 samples. Each set contains pixel data as well as its label in separate arrays. These samples will be used to evaluate the performance of the machine learning model after it has been trained.

Next, we create a list named digit__samples to store samples of each digit. It is a list of arrays, where each array contains all the images of a particular digit from the training set. This structure allows for easy access to all images of a specific digit for further analysis or processing.

A heatmap is a graphical representation of data where values in a matrix are represented as colors. It is a way to visualize data in a 2D format  with each cell in the heatmap colored according to its value. For each digit (0 through 9), we will sum the pixel values of all images corresponding to that digit in the training set. This results in a single vector or array representing the total pixel values for that digit. Each element of this vector represents the sum of pixel values at a particular position (i.e., the position of the pixel in the original images).Now once we have these vectors representing the sum of pixel values for each digit, we can visualize them using a heatmap. In the heatmap, each cell represents a pixel position, and the color of the cell corresponds to the value of the sum of pixel values at that position. Higher values are typically represented by brighter or warmer colors (eg red or yellow)  while lower values are represented by darker or cooler colors (eg blue or green). When we plot the heatmap for each digit, we get the optimal shape of each digit as expected.

Now we need to predict the digits using information theory tools. First, we need to find the probability of labels of each digit (class labels)  occuring in the training set. Then we calculate the entropy which is a measure of uncertainty for these probabilities. The entropy is calculated using the formula:

$$
H(X) = -\sum_{i=1}^{n} P(x_i) \log_2(P(x_i))
$$

We then estimate the mutual information between each pixel (X) and the class label (Y). Mutual Information is a measure from information theory that quantifies the amount of information obtained about one random variable through observing the other random variable. It measures how much knowing the value of one variable reduces uncertainty about the value of another variable. It can be calculated using this formula:

$$
I(X;Y) = H(Y) - H(Y \mid X=0) \cdot P(X=0) - H(Y \mid X=1) \cdot P(X=1)
$$

where H(Y) is the uncertainity in class labels, H(Y|X=0) and H(Y|X=1) are the conditional entropies that measure the uncertainty in the class labels given the value of the pixel (either 0 or 1). P(X=0) and P(X=1) are the probabilities of the pixel taking the value 0 or 1. The reason for calculating the mutual information is to find out which pixels are most informative for predicting the class label. A higher mutual information between a pixel and the class label means that knowing the value of that pixel provides more information about the class label, making that pixel more important for the classification task.

Also we evaluate how accurately each pixel can predict the digit labels. To do this, we look at each pixel individually and check how often its value correctly predicts the actual digit label. This includes both true positives (where the pixel and the label both represent the same digit) and true negatives (where the pixel and the label both correctly indicate the absence of a certain digit). We calculate the prediction accuracy of a pixel by dividing the number of its correct predictions by the total number of images in the dataset. This gives us a measure of how well each pixel can predict the digit labels, with a higher value indicating better predictive power. Interestingly, we find that pixels with high prediction accuracy are often located in the middle of the image. This is because the middle area of the image tends to contain distinctive features that vary between different digits, making these pixels more informative for prediction.


Next we will move on to the main part where we will test the predictive power of pixels that have high mutual information with the labels using the test data. First, we will sort the pixels in descending order based on their mutual information with the class label. We then select the top ‘m’ pixels, where ‘m’ ranges from 1 to 90, and calculate the mutual information between these selected pixels and the class labels.

In our plot, the plot reveals that after selecting 40 pixels, we reach the maximum entropy, which is almost equal to the entropy of the class label.
When we select the top 40 pixels based on their mutual information with the class labels, we are selecting the pixels that provide the most information about the labels. As we add more of these informative pixels, the entropy of the labels given these pixels decreases because these pixels help reduce the uncertainty about the labels.

But after a certain point (here it's 40 pixels), adding more pixels doesn’t significantly reduce the entropy anymore. This is because the additional pixels are not as informative as the top ones. Therefore they don’t contribute much to reducing the uncertainty about the labels. At this point, we say that we have reached the maximum entropy, which is almost equal to the entropy of the class labels. So this observation suggests that the top 40 pixels are sufficient to capture most of the information about the labels in the MNIST dataset. Adding more pixels beyond this point does not significantly improve the predictability of the labels.

Next, we use only the top ‘m’ pixels, which have the highest mutual information with the class, to predict the class labels. These pixels, due to their high mutual information, could potentially provide a pattern to classify the images.

We construct a classifier for a range of ‘m’ from 1 to 90, which includes only these ‘m’ pixels. We then apply this classifier on the test dataset. If we can find a pattern using only these pixels, we have successfully predicted the label. The plot of the prediction accuracy for these ‘m’ pixels shows an interesting trend. The accuracy increases from the single pixel accuracy at 0.215 until we reach a maximum accuracy of 0.53, and then it drops to zero, significantly below the single pixel accuracy. This suggests that there’s a limit to finding patterns with high mutual information that can fully predict the labels. This is because there are pixels with low mutual information which are used by the network, and therefore not present in our selected pixels. While the maximum prediction accuracy is not considered a high value, it still demonstrates the significant predictive value of mutual information between data and labels.

So in conclusion, Information theory tools provide a powerful means to identify and classify objects within images that enhancing our understanding and interpretation of complex visual data.

## Output
### Samples of binarised MNIST dataset
<div style="display: flex; justify-content: center;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/aa165f8b-1dbd-4a0e-9758-ba096890ec83" style="margin: 5px; width: 200px;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/4566e2a7-735a-4015-882b-8cf1c053d81b" style="margin: 5px; width: 200px;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/aaa305d0-9cc2-46e0-a8e7-c3abe8201da7" style="margin: 5px; width: 200px;">
</div>

<div style="display: flex; justify-content: center;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/bd3d459a-13b7-4874-bb68-58de31ebe913" style="margin: 5px; width: 200px;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/784413b3-7bf2-43c0-9bea-2d87b2cfdb16" style="margin: 5px; width: 200px;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/931da2d0-6abd-48fc-9ea6-3d05bc28bb96" style="margin: 5px; width: 200px;">
</div>

<div style="display: flex; justify-content: center;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/528c4885-4d6a-4395-a864-179dcdb7bdad" style="margin: 5px; width: 200px;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/fc19e706-4e6d-40cc-aa2a-e2319e82efd3" style="margin: 5px; width: 200px;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/afa1845b-4a60-43fc-8910-70aa5e4d3482" style="margin: 5px; width: 200px;">
</div>
<div style="display: flex; justify-content: center;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/21c61fc4-742f-42b4-865d-49917ca481f8" style="margin: 5px; width: 200px;">
</div>

### Digit pixel heatmap

<div style="display: flex; justify-content: center;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/0ab93ce9-be28-4c5d-a2df-bbcf2fa2230a" style="margin: 5px; width: 200px;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/74cfd940-603f-4509-b838-79ab15008dee" style="margin: 5px; width: 200px;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/f10f0ada-6aa8-48a6-ab20-f3e5d9ba3153" style="margin: 5px; width: 200px;">
</div>

<div style="display: flex; justify-content: center;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/2ab7ecae-247d-4d80-a50d-367f406eede9" style="margin: 5px; width: 200px;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/865bad63-3204-44a2-ba9e-1496d49a7b1c" style="margin: 5px; width: 200px;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/cdaff843-de61-44d2-bc7d-01a8015ff904" style="margin: 5px; width: 200px;">
</div>

<div style="display: flex; justify-content: center;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/e9b17a69-9003-4f27-b77a-eb8cb960e95c" style="margin: 5px; width: 200px;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/0673add2-c06e-4d7a-ac9d-ae9f108b4496" style="margin: 5px; width: 200px;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/e871c895-c1a1-4db6-ae81-8cf4d3d349ab" style="margin: 5px; width: 200px;">
</div>
<div style="display: flex; justify-content: center;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/c49805b9-b38b-4833-a256-4ed2ceea05ac" style="margin: 5px; width: 200px;">
</div>

### Prediction accuracy of each pixel on training data
<img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/80bcb315-a492-49f9-b9c8-c5788f399fa7" style="margin: 5px; width: 700px;">

### Prediction accuracy of each pixel on training data
<img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/829d1a19-d9bb-48d9-82f9-ecd3407f0bcc" style="margin: 5px; width: 700px;">

### I(X;Y) differentiated by pixel accuracy and normalised pixel accuracy

<div style="display: flex; justify-content: center;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/f6cb486b-8559-4434-b1e2-d9e69ca3e51e" style="margin: 5px; width: 500px;">
    <img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/37f08f24-3968-43c4-8f5b-4ff7fc9db676" style="margin: 5px; width: 500px;">
</div>

### Top ’m’ pixels, Number of pixels selected V/s I(X;Y) with class label
<img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/9cc3be3d-dee8-416a-a79f-c148861e42dc" style="margin: 5px; width: 700px;">

### Prediction accuracy of top ’m’ high I(X;Y) pixels
<img src="https://github.com/FasnaSharaf/Information-Theory-Analysis-of-Supervised-Learning-using-MNIST-dataset/assets/83363902/16ae728b-b33b-419d-ab0a-fccd91cd5fd0" style="margin: 5px; width: 700px;">








