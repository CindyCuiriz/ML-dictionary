# ML-dictionary

## WHAT IS MACHINE LEARNING?

Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed.
It is the science (and art) of programming computers so they can learn from data.

## MACHINE LEARNING CATEGORIES? 

#### Supervised
* k-Nearest Neighbors
* Linear Regression:  You feed it your training examples and it finds the parameters that make the linear model fit best to your data. This is called training the model. 
* Logistic Regression 
* Support Vector Machines (SVMs) 
* Decision Trees and Random Forests • Neural network
#### Unsupervised
* Clustering
  * K-Means
  * DBSCAN
  * Hierarchical Cluster Analysis (HCA)
*  Anomaly detection and novelty detection
  * One-class SVM
  * Isolation Forest
* Visualization and dimensionality reduction
  * Principal Component Analysis (PCA)
  * Kernel PCA —Locally-Linear Embedding (LLE)
  * t-distributed Stochastic Neighbor Embedding (t-SNE)
* Association rule learning
  * Apriori
  * Eclat
#### Semisupervised

Algorithms that can deal with partially labeled training data, usually a lot of unlabeled data and a little bit of labeled data. Google photos is a great example (photo recognition and labeling)

#### Reinforcement Learning

An agent can observe the environment, select and perform actions, and get rewards in return (or penalties in the form of negative rewards).  
It must then learn by itself what is the best strategy, called a policy, to get the most reward over time. A policy defines what action the agent should choose when it is in a given situation.
For example, it is used on robots so they learn how to walk. 

#### Online vs batch learning

* Batch Learning:  The system is incapable of learning incrementally: it must be trained using all the available data. This will generally take a lot of time and computing resources, so it is typically done offline. First the system is trained, and then it is launched into production and runs without learning anymore; it just applies what it has learned. This is called offline learning. If you want a batch learning system to know about new data (such as a new type of spam), you need to train a new version of the system from scratch on the full dataset (not just the new data, but also the old data), then stop the old system and replace it with the new one.
* Online Learning: You train the system incrementally by feeding it data instances sequentially, either individually or by small groups called mini-batches. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives. It is great for systems that receive data as a continuous flow (e.g., stock prices) and need to adapt to change rapidly or autonomously.

## One important parameter of online learning systems is how fast they should adapt to changing data: this is called the learning rate. If you set a high learning rate, then your system will rapidly adapt to new data, but it will also tend to quickly forget the old data (you don’t want a spam filter to flag only the latest kinds of spam it was shown). Conversely, if you set a low learning rate, the system will have more inertia; that is, it will learn more slowly, but it will also be less sensitive to noise in the new data or to sequences of nonrepresentative data points (outliers). A big challenge with online learning is that if bad data is fed to the system, the system’s performance will gradually decline. If we are talking about a live system, your clients will notice. For example, bad data could come from a malfunctioning sensor on a robot, or from someone spamming a search engine to try to rank high in search results. To reduce this risk, you need to monitor your system closely and promptly switch learning off (and possibly revert to a previously working state) if you detect a drop in performance. You may also want to monitor the input data and react to abnormal data (e.g., using an anomaly detection algorithm). 


#### Instance based vs model-based learning
This way of categorizing ML algorithms is by the way they generalize examples they have seen before. A good performance on the training data is insufficient, the true goal is to perform good on new instances/predictions.

* Instance based learning:  the system learns the examples by heart, then generalizes to new cases by comparing them to the learned examples (or a subset of them), using a similarity measure.

* Model-based learning:  Build a model of these examples, then use that model to make predictions

## UTILITY FUNCTION / FITNESS FUNCTION

Measures how good your model is

## COST FUNCTION

Measures how bad is your model. This one is mainly used for linear regression problems.

## DIMENSIONALITY REDUCTION

Simplify the data without losing too much information. One way to do this is to merge several correlated features into one

## FEATURE EXTRACTION

For example, a car’s mileage may be very correlated with its age, so the dimensionality reduction algorithm will merge them into one feature that represents the car’s wear and tea

## ANOMALY DETECTION

Identifying data points in data that don't fit the normal patterns. For example, detecting unusual credit card transactions to prevent fraud, catching manufacturing defects, or automatically removing outliers from a dataset before feeding it to another learning algorithm

## NOVELTY DETECTION

Statistical method used to determine new or unknown data and determining if these new data are within the norm (inlier) or outside of it (outlier).

## ASSOCIATION RULE LEARNING

The goal is to dig into large amounts of data and discover interesting relations between attributes.

## MAIN ML CHALLENGES

### BAD ALGORITHM OR BAD DATA

## SAMPLING BIAS

It is crucial to use a training set that is representative of the cases you want to generalize to. This is often harder than it sounds: if the sample is too small, you will have sampling noise (i.e., nonrepresentative data as a result of chance), but even very large samples can be nonrepresentative if the sampling method is flawed.

## POOR-QUALITY DATA

It is worth the effort to spend time cleaning up your training data. 

* If some instances are clearly outliers, it may help to simply discard them or try to fix the errors manually. 
* If some instances are missing a few features (e.g., 5% of your customers did not specify their age), you must decide whether you want to ignore this attribute altogether, ignore these instances, fill in the missing values (e.g., with the median age), or train one model with the feature and one model without it, and so on

## IRELEVANT FEATURES

 Your system will only be capable of learning if the training data contains enough relevant features and not too many irrelevant ones. A critical part of the success of a Machine Learning project is coming up with a good set of features to train on. This process, called feature engineering, involves:
* Feature selection: selecting the most useful features to train on among existing features.
*  Feature extraction: combining existing features to produce a more useful one (as we saw earlier, dimensionality reduction algorithms can help).
*   Creating new features by gathering new data. 

## OVERFITTING TRAINING DATA

Overgeneralizing, this will make our model too precise for every instance and use patterns that may show a trend but not really be related one to another, causing it to be really impacted by noise.
Complex models such as deep neural networks can detect subtle patterns in the data, but if the training set is noisy, or if it is too small (which introduces sampling noise), then the model is likely to detect patterns in the noise itself. Obviously these patterns will not generalize to new instances.

## HOW TO SOLVE OVERFITTING

Overfitting happens when the model is too complex relative to the amount and noisiness of the training data. 
The possible solutions are:
* REGULARIZATION: To simplify the model by selecting one with fewer parameters (e.g., a linear model rather than a high-degree polynomial model), by reducing the number of attributes in the training data or by constraining the model 
* To gather more training data
*  To reduce the noise in the training data (e.g., fix data errors and remove outliers)

## REGULARIZATION
It is the process of making a model simpler, it can mean adjusting your parameters to find the right balance between fitting the training data perfectly and keeping the model simple enough to ensure that it will generalize well. (degrees of freedom could be adjusted in this instance to the mean)

## HYPERPARAMETERS

This is the way we control regularization to apply during learning. A hyperparameter is a parameter of a learning algorithm (not of the model). It is not affected by the learning algorithm itself; it must be set prior to training and remains constant during training. If you set the regularization hyperparameter to a very large value, you will get an almost flat model (a slope close to zero); the learning algorithm will almost certainly not overfit the training data, but it will be less likely to find a good solution. Tuning hyperparameters is an important part of building a Machine Learning system

## UNDERFITTING THE TRAINING DATA

It occurs when your model is too simple to learn the underlying structure of the data. 

## HOW TO SOLVE UNDERFITTING

* Selecting a more powerful model, with more parameters
*  Feeding better features to the learning algorithm (feature engineering)
*   Reducing the constraints on the model (e.g., reducing the regularization hyperparameter)

## TESTING AND VALIDATING

You can and should test your model before you put it in production, to do so we can plit our data in two sets, the training set and the test set, the error rate on new cases is called the GENERALIZATION ERROR/ OUT OF SAMPLE ERROR, by evaluating your model on the test set, you get an estimate of this error. This value will tell you how well your model will perform on instances it has never seen before.

LOW TRAINING ERROR (Your model makes few mistakes on the training set) & HIGH GENERALIZATION ERROR = MODEL IS OVERFITTING THE TRAINING DATA

### COMMON PRACTICE IS TO USE 80% OF THE DATA FOR TRAINING AND 20% FOR TESTING
### THIS ALSO DEPENDS ON THE SIZE OF THE DATASET, FOR EXAMPLE, A DATASET WITH 10 MILLION INSTANCES IT IS MORE THAN ENOUGH USING 1% FOR TESTING, WHICH EQUALS 100,000 INSTANCES

## HYPERPARAMETER TUNING AND MODEL SELECTION

Sometimes you want to compare different models to see which one is better, to do so you train them and compare how well they generalize using the test set, After that you get to a model that generalizes better, and then you need to apply some regularization to avoid overfitting, to do so you need to choose a regularization hyperparameter. To do so you can again train  100 different models using 100 different values for this hyperparameter.
 ### Where is the problem? 
 The problem is that we measured the generalization error multiple times on the test set, and you adapted the model and hyperparameters to produce the best model for that particular set. This means that the model is unlikely to perform as well on new data. 

### How to solve it? 

























