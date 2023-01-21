# Kaggle-Competition-Plant-Seedlings-Classification-Resnet18

Capstone Project
Plant Seedlings Classification
By Samy Mohsen
https://www.kaggle.com/competitions/plant-seedlings-classification/


Project Specifications:

1-The Notebook coded on Kaggle platform.  
2-Language Python.
3-cpu used for writing the entire notebook.
4-cuda instance used for training and evaluating.
5- Accelerator used : GPU T4 x2 , RAM 13 GB.

Project Overview:
1-Domain Background:
The Aarhus University Signal Processing group, in collaboration with University of Southern Denmark submitted Plant Seedlings Classification as a competition on Kaggle. The approaches used range from weed maps constructed using coarse remote sensing data to real-time precision spraying using ground-based platforms equipped with high-resolution imagery sensors. Common to all approaches is the goal of detecting weeds - either in patches or as single plants. Although some systems are commercially available, but such great system with better results will become a true commercial breakthrough for the agricultural industry.
For detailed information:  https://arxiv.org/abs/1711.05458
2-Problem Statement:
As it requires in the competition description: Can you differentiate a weed from a crop seedling? Botanists have been dealing with species categorization for centuries and substantial progress is reported in the field of content-based image retrieval and analysis of images and video. The present authors believe that one problem is a lack of benchmark databases. Several studies on species recognition contain a description of preprocessing steps such as image acquisition, segmentation and annotation which suggest that researchers have spent time on these topics although each of these tasks is an area of its own. To support and encourage the development of species recognition techniques for the agricultural industry. So, the authors present a database that is freely available to researchers and which enables them to jump directly to the task of object analysis, species recognition or plant appearance analysis, and the ability to do so effectively can mean better crop yields and better stewardship of the environment.

Evaluation Metrics:
The project will be evaluated on MeanFScore, which at Kaggle is actually a micro-averaged F1-score.
Given positive/negative rates for each class k, the resulting score is computed this way:

$$Precision{micro} = \frac{\sum{k \in C} TPk}{\sum{k \in C} TPk + FPk}
Recall{micro} = \frac{\sum{k \in C} TPk}{\sum{k \in C} TPk + FNk}
MeanFScore = F1{micro}= \frac{2 Precision{micro} Recall{micro}}{Precision{micro} + Recall_{micro}}$$

That the metric that Kaggle will be evaluating my project according to compare it with the benchmark model and the results of predicted classes(species) with the real one. 

Benchmark Model:                            (Quoted From the Author Scientific research Paper Here)
With common public data material as presented in this paper, researchers are one step closer to being able to compare results. Yet another step would be to agree on a performance measure. We propose to use measures derived from classification results achieved by stratified cross-validation of any preferred classification algorithm. The proposed procedure is as follows. The data to be used is shuffled and divided into a number of disjoint sets (folds). The classifier under test is trained with data from all but one of these folds, and its performance after training is evaluated with the remaining one. Performance is measured using weighted averages of f1 scores for each fold, where the weighting should be according to class sample size. This procedure results in one performance measure for the classifier per fold. Now assume that the set of weighted average f1 score, Sf , constitutes a set of samples from a population of possible performance results for the classifier under test. (The f1 score is a commonly used measure (Lu and Wang, 2009) and is invariant against sample size and the ability of a classifier to recognize true negatives (Sokolova and Lapalme, 2009).) This measured performance is a random variable since the data were selected randomly by shuffling. We perform bootstrapping of the set Sf , to generate alternative possible sample sets from the same population, and generate 1000 bootstrap sets. Calculating the average of each bootstrap set produces new samples of the underlying population of performance measurements. Generating these samples enables the calculation of confidence intervals. One simple way of doing this would be to sort the bootstrap set averages into ascending order, Lavg, and extract, for the 95% confidence interval, values at the indices 0.05 ∗ 0.5 ∗ 1000 = 25 and 1000 − 0.05 ∗ 0.5 ∗ 1000 = 975. The proposed measures for reporting classifier performance are then the fold average of weighted average f1-score and its associated confidence interval achieved by bootstrapping. We recommend using 10 fold cross validation.
 
where T Pc, F Pc and F Nc denotes True positives, False positives, and False negatives for class c respectively. Pc is class specific precision and Rc is class specific recall. N denotes the total number of samples and Nc the number of samples of class c and C the total number of classes. S is the mean of Av weighted(f1) across all cross-validation folds. The f1-score could be used for comparing both segmentation algorithms and specie classification algorithms. Assessing segmentation algorithms will, however, be difficult since no ground truth is available. For specie classification, the necessary annotation information is present and calculating f1- scores is straightforward. We therefore only propose a benchmark for classification here, though we plan to extend the database with annotations of “correct” segmentations which would facilitate the definition of a segmentation benchmark.



Project Steps:

1- Getting data:
Since the competition on Kaggle with the data that the author provided, and I’m working o Kaggle platform. I used the code that Kaggle was default provided it at the start of the notebook.

The dataset containing images of approximately 960 unique plants belonging to 12 species at several growth stages.  
The dataset comprises 12 plant species. 
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet


2- EDA

The dataset Contains 2 folders and sample_submission.csv file.
The train folder is the labeled dataset that will be used for training and validation, the other folder contains unlabeled images that will be used as inferences for the model to predict the species and include it to the submission file to evaluate my model by Kaggle metrics. 

The test data set (prediction) contains of 794 images.

I took the data set and made a data frame using pandas, and it contains 4750 image and 12 unique values (classes) with no NaN values. I also used groupby to see the distribution of the images and it returns that it’s not equally distributed for the classes. 
When started to get samples for images to explore visually. I recognized that the images are different in shape and size.
 
2-Build model:
Now I started to create our testing function to evaluate my model with testing loop to calculate the average loss and the accuracy and to get idea about how my model preform. Then the train function that will be used to train our model included training loop and calculating the accuracy and the loss of our training process. 
Then created the model function using the pretrained model resnet18 because I’m more familiar with plus I know it’s hyperparameter. Making the output 12 according to our classes’ numbers. Using Cross-Entropy Loss for cost function since its multiclass classification and using Adam for optimization learning rate = 0.001.

3- Preparing data:

First, we will split our labeled training data set to train data set and test (validation) data set by ratios 80%, 20% it will be 3800 image for training, 950 image for validation.
Second we will make our transforms to the images to be the same size and shape, size it to (224,224), and normalizing by [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].

4- Start training and evaluating.
We will start to run our functions and get the results to analyze and see how our model preform. Setting our hyperparameters to be batch size = 32 and our epoch to 5.

5-Analyiz results:

My first attempt to train with epoch 5 and batch size 32

  
As I see the accuracy increasing with increasing the numbers of epoch and the loss is decreasing also that means with tuning of hyperparameters will lead to better result, and the model performance will increase.


My final attempt it was batch size 64 and epoch 10
 


 

As I noticed that increasing the epoch numbers make a slightly different results that improved the accuracy and lower the loss. Also, it reflected on my test set (validation) the accuracy increased to 75%.
5- Changes on the model to improve results:               (Added for Resubmission)
My first try my hyperparameters was set to be epoch 5 and the batch size 32 learning rate was 0.001. The training accuracy on the last epoch (5) was 74% and my submission score was .1020. but when I saw the curve of the training accuracy increases with increased number of epochs, I decided to increase the epochs hoping to increase the result score.

My 2nd attempt was tuning the hyperparameters to be the batch size 32 and the epoch set to be 15 learning rate 0.001, remaining the same data processing steps and model architecture. my submission score was .1032.

My final attempt was sitting the hyperparameter batch size to 64 and the epochs to 10 to avoide long training time, learning rate 0.001, remaining the same data processing steps and model architecture. And the results were 78% training accuracy, and 75% on validation (test set) and submission score was .1039.

6- Predictions for Kaggle submission:

First created csv file contain the test image as Kaggle requires, contains the path and name of every file from the 794 files they provide. Then created a new data frame from this test.csv to start my prediction.
After saving the model and calling back the model to cpu to start the prediction. Tried first predict single image take the URL and do image transformers used test_transformers_pred. , then calling it’s class using .argmax() then comparing it to the class list to get the species. 

note:(anyway as I understand that Kaggle don’t save the model, every time I end the session it deletes it and I need to repeat all steps from the beginning once again… my thoughts for that because the training done on the same processor as the notebook run , not like sagemaker you can run the note book in instance you choose and do the training job on another instance,, really not sure )

Then done a loop to take the past process and take all the file paths from my test(inferences) dataframe, and store the result in list to be able to make another dataframe contains the species. Changing all the columns in the dataframes to str format to be able to merge. Then with merge, merging the 2 dataframes together, drop the (path) column and save the new data frame.

Now I have a dataframe , that contains all the file names from the test (prediction) set and other column that species that my model predicted … save this to (submission.csv).

7-Kaggle submission:

 
The first score was .102 from the first submission, when was the batch size 32 and the epoch 5 .

 

The final score was .1039, that when the batch size 64 and the epoch 10.




8- Justification, Comparing my model to benchmark model.
The benchmark model proposes to use measures derived from classification results achieved by stratified cross-validation of any preferred classification algorithm which we did in our model we used method that uses different portions of the data to test and train a model on different iterations. Also, the benchmark requires the data to be used is shuffled and divided into a number of disjoint sets (folds). The classifier under test is trained with data from all but one of these folds, and its performance after training is evaluated with the remaining one, which we did in our data, the data was shuffled and split to training and testing, the training done on all the data except the test set was used only to evaluate. Also, the benchmark model measure performance by using weighted averages of f1 scores, same as the evaluation metric that Kaggle use to compare model accuracy with the benchmark model, and it is calculated from the precision and recall of the test, where the precision is the number of true positive results divided by the number of all positive results, including those not identified correctly, and the recall is the number of true positive results divided by the number of all samples that should have been identified as positive and since F1-score scales (from 0 to 9, 0 being lowest and 9 being the highest) and my final model f1 score is .1039 that means my model is not significant enough to have adequately solve this problem.

9- Note for myself to improve: 
The model can be much better, with better results on Kaggle f1 score and better prediction accuracy. I’m saying that because changing a little from the hyperparameters changed the score slightly better. If this project was done on AWS platform, using sagemaker and the tools that AWS provide, variety of training instance, using sagemaker debugger profiling report. It will be a great tool to get more information how my model performs, what errors triggered, get some info about gpu utilization, bottlenecks...etc.
Plus, changing hyperparameters slightly increased the results I have. I could make multi training jobs with sitting ranges for hyperparameters like batch size, and learning rate and let sagemaker get the best estimator, it will affect the performance and the accuracy of my model. Of course will be in mind the cost and the resources available.

Also could use another pretrained model or dd improvements to my data processing steps

Sorry for the long report but I wanted to report all the process with details.
Regards,
Samy Mohsen
