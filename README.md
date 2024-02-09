# CSE151A-MachineLearningFinalProject
CSE151A-MachineLearningFinalProject

https://colab.research.google.com/github/varunraisinghal/CSE151A-MachineLearningFinalProject/blob/main/CSE151A_Machine_Learning.ipynb

## Preprocessing Process
### Things Already Done
1. First, we loaded the CSV file and converted it to a pandas Dataframe for ease of manipulation.
2. Next, after noticing missing values within the `in_shazam_charts` and `key` feature columns, we decided to do two things. First, we replaced the `in_shazam_charts` NA values with `0` since if the value for `in_shazam_charts` is not applicable, then it must not be in the Shazam charts, so it would in turn be false (aka `0`). Second, we decided, for now, to completely drop the `key` column since there were 95 missing values which will likely detract from our model without imputation.
3. We used the .describe() method to provide a summary of the numerical columns in the dataset. We wanted to do this to get a good understanding of the basic statistical details like mean, standard deviation, minimum, and maximum values of each numeric feature so that we can get a good sense of our data.
4. We also created histograms for all the various features to help understand the distribution of the characteristics of the songs. Heatmap was created as well as a pairplot in order to identify trends and analyze how different features correlate to each other. 
### Things To Be Done
1. Depending on the determined necessity, we may impute the `key` column missing values if the inclusion of the `key` column improves our model's performance.
2. After determining the most relevent features to our model, which as of now are the characteristic percentages of the songs \[`danceability_%`, `valence_%`, `energy_%`, `acousticness_%`, `instrumentalness_%`, `liveness_%`, `speechiness_%`\], we will isolate those features from our dataset.
3. Then, we will divide our relavent dataset into separate training and testing datsets using a 80/20 random split. We may modify the proportions if we recognize signs of over or under fitting.
4. Since currently all our features are percentages that range from 0 to 100, we will not have to standardize our dataset. However, if we decide to include other features, we may have to standardize our values.
5. Since every song name is unique and non-numeric, we may have to assign each song a unique identifier integer ranging from 1 to 953.
