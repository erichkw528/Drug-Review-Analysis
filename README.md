# Drug-Review-Analysis
## Initial Goals
Our approach employs both supervised (logistic regression) and unsupervised models(k-means clustering) to determine correlations between various attributes of the Drug review dataset. Our main goal is to accurately predict how a drug would be rated based on a given review. Furthermore, we aim to discover patterns and trends in the data that can effectively guide decision-making in future drug recommendations.


## The Overarching Approach
Our consensus is to devise a system for discerning the differences between good, average, and bad ratings. After discussion we conclude that the optimal course of action involves grouping each review into a category ranging from 1 to 10 given by the rating column in our data set. We decided to categorize the ratings of 1 to 4 to be marked as ‘bad’, while ratings of 5 to 7 to be marked as ‘average’, and ratings of 8 to 10 to be marked as ‘good’. 

## The Technical Approach
First, we clean up the text reviews by removing unnecessary words(like in/or/that) that won't help us classify the reviews. Then we use a method called TF-IDF to classify the remaining words as either positive, neutral, or negative. Finally, we use a logistic regression model that looks at the reviews and predicts which category they belong to based on the words used. We compare the predictions to the actual ratings from a testing dataset to see accuracy of the model.

## Dataset 
Our dataset consists of over 309,000 drug reviews retrieved from the UCI Machine Learning repository. The dataset provides patient reviews on drugs paired with related conditions and a rating system to reflect patient satisfaction. The 6 attributes in this dataset are drugName (name of the drug), condition (name of condition), review (patient text review), rating (a scale from 1-10), date (the date the review was entered), and usefulCount (number of patients that found the review useful). We utilized drugName, review, usefulCount, rating, and condition to perform our tasks. The date column was only used for our time-series.

Link: https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29# 

## Data Cleaning
The following code is preparing the 'date' column of the dataframe. It does this by first importing the datetime module that helps manipulate date and time data. It then converts the 'date' column of the DataFrame from a string format(May 17, 2009) to a datetime format(05/17/2009) using pandas.to_datetime.
Next, we rename the 'Unnamed: 0' column in the original dataset to 'rating_id' using the pandas rename method. Finally, it formats the 'date' column to a specific date format 'MM/DD/YYYY.’ This standardizes the date format of the 'date' column and makes it easier to analyze or visualize.

#### Here is the data after initial cleaning:
<p align="center">
  <img width="626" alt="data after initial cleaning" src="https://github.com/erichkw528/Drug-Review-Analysis/assets/97138813/fa86e2f5-aa40-41f7-b8b1-7e3a51f95480">
</p>

## Visualizations + ML Models

### 1. Distribution of drug ratings
<p align="center">
  <img width="621" alt="Distribution of Drug Ratings" src="https://github.com/erichkw528/Drug-Review-Analysis/assets/97138813/58e892e0-bbc8-4d90-b133-b366facbd5b8">  
</p>

<p align="center">
  Figure 1: Distribution of drug ratings
</p>

**Line graph**: The line graph shows us the distribution of our drug ratings (x-axis) in relation to number of reviews or frequency (y-axis). We can see that there are more occurrences of highly rated reviews than lower. The line graph of drug ratings showing more highly rated reviews than lower ones could be due to several factors, including selection bias or the drug's actual effectiveness and side effects compared to other drugs. If the reviews are only from people who chose to review the drug, positive experiences may be overrepresented. If the drug is objectively more effective or has fewer side effects than other drugs, this could result in more positive reviews. We considered these factors and approached the interpretation of the graph with caution to draw accurate conclusions. 

**Pie Chart**: The pie chart allows us to visualize the distribution of drug ratings in a drug review dataset by displaying the proportion of each rating value as a slice of the total pie. It provides an additional illustration to help understand the distribution of ratings in our dataset. 

### 2. Top 10 drugs by review count:
<p align="center">
  <img width="485" alt="Top 10 Drugs by Review Count" src="https://github.com/erichkw528/Drug-Review-Analysis/assets/97138813/79aa066b-3ec4-46cf-91ac-94c4a26de7db">
</p>

<p align="center">
  Figure 2: Bar chart for the Most Mentioned Drugs
</p>

This bar chart shows us the top 10 reviewed drugs. The drugs are ordered in descending order from the most reviewed to the least. From this chart, we can conclude that the most common types of drugs in this dataset are ones treating birth control (based on a quick google search of the drugs listed in the graph above). 

### 3. Top 10 most frequent conditions :

<p align="center">
  <img width="472" alt="Top 10 Most Frequent Conditions" src="https://github.com/erichkw528/Drug-Review-Analysis/assets/97138813/ab965382-0029-4831-8bf9-6b006e67a75b">
</p>

<p align="center">
  Figure 3: Horizontal Bar Chart Most Common Conditions
</p>

This bar chart shows us the top 10 most frequent conditions. We can see that birth control is the most common condition in this dataset. This is also confirmed from Figure 2, where the most common types of drugs in the dataset were found for birth control.  

### 4. Relation between ratings and usefulness of a review:
<p align="center">
  <img width="334" alt="Relationship between ratings and usefulness of a review" src="https://github.com/erichkw528/Drug-Review-Analysis/assets/97138813/144f4853-e44c-4d10-a9bd-312ee2e76421">
</p>

<p align="center">
  Figure 4: Scatter Plot
</p>

This scatter plot shows us the distribution (with dots) of our useful count (x-axis) in relation to each rating (y-axis), where each point represents a rating and its corresponding useful count. This helps us observe any correlation between these variables. We can see that higher ratings tend to have more useful counts than lower ratings.

### 5. Word Cloud of most appearing words in reviews:
<p align="center">
  <img width="513" alt="Word Cloud of most appearing words in reviews" src="https://github.com/erichkw528/Drug-Review-Analysis/assets/97138813/3c8d189f-747d-4a55-ad58-5363bf1628af">
</p>

<p align="center">
  Figure 5: Word Cloud
</p>

This shows us the distribution of words in the reviews column in relation to how often it was used in each category. This gives an idea of the most common complaints or compliments patients have about the drugs. This word cloud provides a visual description of the most commonly used words in the reviews of our dataset.

## Trends Visualization (Time Series) 
<p align="center">
  <img width="622" alt="Frequency of Top Drug for Each Year" src="https://github.com/erichkw528/Drug-Review-Analysis/assets/97138813/5f145450-6502-4499-a1aa-ee42c5814d1a">
</p>

Time series data is a representation of changes in a particular phenomenon over time. In the above chart, we employed the 'year' variable from our dataset to identify the popular contraceptive drug within the timeline. We observed that Levonorgestrel, a pregnancy prevention drug (contraceptive), garnered the majority of reviews. Notably, Ethinyl estradiol / norethindrone, another pregnancy prevention medication, gained prominence in 2010, 2012, and 2013. Similarly, Etonogestrel, another pregnancy prevention pill, was prevalent in 2011 and 2014. However, from 2015 to 2017, Levonorgestrel received the most frequent reviews and experienced exponential growth in popularity compared to the other two contraceptives.

It is important to acknowledge that using quarterly or monthly data rendered our graph cluttered and indistinct. Furthermore, tighter intervals resulted in the appearance of more drug types, thereby diminishing our ability to discern trends.



### Data preprocessing:

For preparing our dataset to be fed into the regression model, we first clean the reviews column.
We use the regular expressions (re) module and the Natural Language Toolkit (nltk) module to manipulate text data, and import the word_tokenize, stopwords, and pos_tag functions from the nltk.tokenize, nltk.corpus, and nltk.tag modules, respectively. 
We then proceed by defining a function called clean_reviews that will be applied to the 'review' column of all instances in the dataframe. This function takes a review string as input, removes special characters and digits, converts the text to lowercase, removes stop words, removes words with less than two characters. Additionally, it also removes words that do not start with adjectives (JJ) and adverbs (RB) from the remaining words using the pos_tag function. 
We then apply the clean_reviews function to the 'review' column of the DataFrame using the pandas library. This will clean each review in the 'review' column and return the cleaned text back to the 'review' column of the dataframe.

Snippet of the dataset after preparing the reviews column:

<p align="center">
  <img width="516" alt="Snippet of the dataset after preparing the reviews column" src="https://github.com/erichkw528/Drug-Review-Analysis/assets/97138813/321dc530-3f87-4386-b07e-952bd8a14e85">
</p>

## TF-IDF (Text mining)  
<p align="center">
  <img width="589" alt="TF-IDF (Text mining)" src="https://github.com/erichkw528/Drug-Review-Analysis/assets/97138813/24e961ac-925e-44a1-9fd9-64ecbe3d0ab8">
</p>

This code above is to predict the sentiment of drug reviews as either good, average, or bad using the TF-IDF vectorization and logistic regression. The code imports the necessary libraries from scikit-learn, such as TfidfVectorizer, accuracy_score, classification_report, and LogisticRegression. These libraries provide tools for vectorizing text data, training and evaluating classification models, and generating reports. The drug reviews and their corresponding sentiment categories are loaded from the dataset into variables reviews_train and categories_train for training the model. Similarly, the drug reviews and their sentiment categories are loaded into variables reviews_test and categories_test for testing the model. A TfidfVectorizer object is created to transform the text reviews into numerical feature vectors. The fit_transform method is called on the training data to create a vocabulary and transform the reviews into a matrix of features. The transform method is then called on the test data to transform the reviews into a matrix of features using the previously created vocabulary. A logistic regression model is then initialized with the max_iter parameter set to 5000 to avoid convergence issues during training. The fit method is called on the training data to train the model. The predict method is called on the test data to generate predictions using the trained logistic regression model. The accuracy of the model is calculated by comparing the predicted sentiment categories with the actual sentiment categories in the test data, using the accuracy_score function.


## Logistic Regression 
<p align="center">
  <img width="410" alt="Logistic Regression" src="https://github.com/erichkw528/Drug-Review-Analysis/assets/97138813/f3852e68-0043-4146-bf2f-ade6b89fa543">
</p>

Logistic regression is a useful algorithm for analyzing this dataset, as it can provide interpretable results, handle both continuous and categorical independent variables, and efficiently handle missing data. It provides interpretable results that can help understand the relationship between the independent variables and the dependent variable. Logistic regression is a computationally efficient algorithm that can handle large datasets with many features. Thirdly, it can handle both continuous and categorical independent variables, making it a versatile algorithm that can be applied to a wide range of datasets. Additionally, logistic regression can handle missing data by using maximum likelihood estimation to estimate the coefficients of the model. It is valuable for analyzing the dataset, as it can provide interpretable results, efficiently handle missing data, and handle both continuous and categorical independent variables. 

With our implementation we were able to predict if a review was good, average, or bad given a text review. The results showed that the model performed exceptionally well in predicting good reviews, with high accuracy, precision, and recall scores. However, the performance metrics of the model decreased significantly when predicting bad reviews, and even more so when predicting average reviews.

Further analysis revealed that the high bias of our model could be a contributing factor to the decreased performance in predicting bad and average reviews. This bias could be due to the limited number of features in our dataset, which hindered the ability of our algorithm to capture the complexity of reviews and the combination of keywords that differentiate between average and bad reviews. Therefore, the model oversimplified the difference between average and bad reviews, resulting in reduced performance for these categories. To improve the performance of our model, we may need to incorporate more features in our dataset to capture the nuances and complexity of reviews. This approach will help to overcome the high bias of our algorithm and enable more accurate prediction of the quality of text reviews.



## K-Means Clustering

K-means is an unsupervised machine learning algorithm that aims to group similar data points into clusters based on their features. 

<p align="center">
  <img width="399" alt="K-Means Clustering" src="https://github.com/erichkw528/Drug-Review-Analysis/assets/97138813/be80702a-96e3-4a08-b9f7-bcfa75892a0d">
</p>

In this case, the code first encodes the categorical variables 'condition' and 'drugName' into binary variables using one-hot encoding. Then, it combines the features into a single matrix where each row represents a drug review, and each column represents a feature (rating, condition, or drug name). 

<p align="center">
  <img width="235" alt="K-Means Cluster Category" src="https://github.com/erichkw528/Drug-Review-Analysis/assets/97138813/114e7110-c817-40c6-b9ae-b7c56d769107">
  <img width="297" alt="Cluster from K-Means Clustering" src="https://github.com/erichkw528/Drug-Review-Analysis/assets/97138813/1a68d11b-62d3-41cd-8261-c3a10db9483e">

</p>

The data is standardized using StandardScaler to have zero mean and unit variance. Next, the code sets the number of clusters to 5 and applies the K-means algorithm using the fit() method. After the model is fit, the code visualizes the clusters using a scatter plot where each point represents a drug review, and the color indicates the cluster to which it belongs. 

<p align="center">
  <img width="267" alt="K-Means Cluster Info" src="https://github.com/erichkw528/Drug-Review-Analysis/assets/97138813/bd163038-e394-4267-a486-46bcdaa2bb36">
</p>

In the end we print out the size of each cluster and the top five reviews for each cluster. This analysis allows us to understand the different groups of reviews based on their features and possibly identify patterns and insights about the drugs and conditions being reviewed.

From the clustering we can observe that the first 2 clusters are bad, the middle cluster is average, and the final 2 clusters are good.

## Agglomerative Clustering
<p align="center">
  <img width="620" alt="Agglomerative Clustering" src="https://github.com/erichkw528/Drug-Review-Analysis/assets/97138813/8d734fd1-0a68-483f-8bc9-8fcee8afb439">
</p>

This code snippet performs Agglomerative Clustering on a subset of our dataset. The subset of data is obtained by randomly sampling 1000 rows from the original dataset as the whole dataset is far too large to do this task on (running into kernel errors). 

Categorical variables in the data are encoded using one-hot encoding, and the resulting feature matrix is standardized using StandardScaler from sklearn. The AgglomerativeClustering algorithm is then applied to the standardized feature matrix to obtain a linkage matrix using the 'ward' method. For this instance, the parameter, distance_threshold, is set to 0, which means that all clusters will be merged into one. Additionally, the default value is set to “None”, which means the algorithm will attempt to find the optimum number of clusters. 

<p align="center">
  <img width="500" alt="Agglomerative Clustering Dendrogram" src="https://github.com/erichkw528/Drug-Review-Analysis/assets/97138813/e88e9439-eedc-4b9b-add6-d844cf932a58">
</p>

A dendrogram is plotted using the linkage matrix to visualize the hierarchical structure of the clusters. The number of clusters is determined based on the dendrogram, and AgglomerativeClustering is applied again to the feature matrix with the chosen number of clusters. The resulting labels for each data point are then used to analyze the reviews within each cluster. Finally, the code prints the size and first few reviews of each cluster. The commented-out code can be used to visualize the clusters using a scatter plot and to calculate the time elapsed for running the code.

<p align="center">
  <img width="302" alt="Agglomerative Clustering Dendrogram Info" src="https://github.com/erichkw528/Drug-Review-Analysis/assets/97138813/7537f6f1-d511-4fc6-b3f7-ad34aab8f8ca">
</p>



## Conclusion + Findings
This project aimed to predict the rating of a drug based on the review given by the user, using a drug review dataset from the UCI machine learning repository. The data cleaning process involved categorizing the data into three categories, followed by applying the TF-IDF method to classify the remaining words as positive, neutral, or negative. The logistic regression model was then used to predict which category the reviews belonged to based on the words used. However, the model's accuracy score of 70% was hindered by the high bias of the model due to the limited number of features in the dataset and the high ratio of good review ratings, which comprised approximately 60% of the entire dataset. The limited availability of bad/average review train data affected the algorithm's ability to capture the complexity of reviews and the combination of keywords that differentiate between average and bad reviews.

Analyzing the trends of the dataset revealed that the majority of drug reviews were on contraceptives. Additionally, there was a clear shift in popularity from Etonogestrel and Ethinyl estradiol to Levonorgestrel, as evidenced by the exponential reviews of Levonorgestrel.

For clustering, the data was first prepared by applying standard scalar and one-hot encoding. Building, training, and testing the logistic regression model took approximately 25 seconds. Both K-means and Agglomerative Clustering took around 7-9 seconds to complete. However, a challenge was encountered when trying to include more than 1000 samples from the dataset for agglomerative clustering, which triggered a kernel error (runtime exceeded). As a solution, 1000 reviews were randomly sampled from the dataset. Conversely, when applying the same number of clusters for k-means, no challenges were encountered. The optimal number of clusters found was n=5 from the agglomerative clustering algorithm.



## Further Documentation / Notes: 

Cleaning reviews cell block took ~293 seconds or approx 5 minutes
Building, training and testing the logistic regression model took ~ 25 seconds.
K-means clustering took 9.027156829833984 seconds to execute.
Agglomerative clustering took 7.622380971908569 seconds to execute.


Important links:

Team Folder link:
https://drive.google.com/drive/folders/1EzTl0Yn4IoVjNW6LDB0Np-bZLDFW9fin?usp=share_link

Presentation link:
https://docs.google.com/presentation/d/1o0Se_obr13M2ERc1CWwjdO8RpJLACl6iVfo2DkKfcjE/edit?usp=sharing

Video link:
https://drive.google.com/file/d/1Vo18M2NdTGfYT0WkgPntXWp9MP5ol2Hs/view?usp=share_link
