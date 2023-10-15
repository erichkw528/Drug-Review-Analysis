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
[insert picture link]


## Visualizations + ML Models

### Distribution of drug ratings
[insert picture link]  
Figure 1: Distribution of drug ratings


**Line graph**: The line graph shows us the distribution of our drug ratings (x-axis) in relation to number of reviews or frequency (y-axis). We can see that there are more occurrences of highly rated reviews than lower. The line graph of drug ratings showing more highly rated reviews than lower ones could be due to several factors, including selection bias or the drug's actual effectiveness and side effects compared to other drugs. If the reviews are only from people who chose to review the drug, positive experiences may be overrepresented. If the drug is objectively more effective or has fewer side effects than other drugs, this could result in more positive reviews. We considered these factors and approached the interpretation of the graph with caution to draw accurate conclusions. 

**Pie Chart**: The pie chart allows us to visualize the distribution of drug ratings in a drug review dataset by displaying the proportion of each rating value as a slice of the total pie. It provides an additional illustration to help understand the distribution of ratings in our dataset. 

### Top 10 drugs by review count:
[insert picture link]

Figure 2: Bar chart for the Most Mentioned Drugs

This bar chart shows us the top 10 reviewed drugs. The drugs are ordered in descending order from the most reviewed to the least. From this chart, we can conclude that the most common types of drugs in this dataset are ones treating birth control (based on a quick google search of the drugs listed in the graph above). 
