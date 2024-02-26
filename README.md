# CSE151A-MachineLearningFinalProject
CSE151A-MachineLearningFinalProject

https://colab.research.google.com/github/varunraisinghal/CSE151A-MachineLearningFinalProject/blob/main/CSE151A_Machine_Learning.ipynb

## Preprocessing Process
### Things Already Done
1. First, we loaded the CSV file and converted it to a pandas Dataframe for ease of manipulation.
2. Next, after noticing missing values within the `in_shazam_charts` and `key` feature columns, we decided to do two things. First, we replaced the `in_shazam_charts` NA values with `0` since if the value for `in_shazam_charts` is not applicable, then it must not be in the Shazam charts, so it would in turn be false (aka `0`). Second, we decided, for now, to completely drop the `key` column since there were 95 missing values which will likely detract from our model without imputation. We will also drop the `mode` column, since it is related to the key and not necessary for objective.
3. Then, we check the datatypes of our feature columns. `in_shazam_charts` is `float64` because of the NA values. Since we have replaced those, we can cast this feature column to `int64`. Another feature column, `streams`, is an `object` when it should be a numeric type. This likely indicates one or more of the entries are faulty, so we can remove that entry and then typecast the feature column as `int64`.
4. We used the `.describe()` method to provide a summary of the numerical columns in the dataset. We wanted to do this to get a good understanding of the basic statistical details like mean, standard deviation, minimum, and maximum values of each numeric feature so that we can get a good sense of our data.
5. We also created histograms for all the various features to help understand the distribution of the characteristics of the songs. Heatmap was created as well as a pairplot in order to identify trends and analyze how different features correlate to each other. 
### Things To Be Done
1. Depending on the determined necessity, we may impute the `key` column missing values if the inclusion of the `key` column improves our model's performance.
2. After determining the most relevent features to our model, which as of now are the characteristic percentages of the songs \[`danceability_%`, `valence_%`, `energy_%`, `acousticness_%`, `instrumentalness_%`, `liveness_%`, `speechiness_%`\], we will isolate those features from our dataset.
3. Then, we will divide our relavent dataset into separate training and testing datsets using a 80/20 random split. We may modify the proportions if we recognize signs of over or under fitting.
4. Since currently all our features are percentages that range from 0 to 100, we will not have to standardize our dataset. However, if we decide to include other features, we may have to standardize our values.
5. Since every song name is unique and non-numeric, we may have to assign each song a unique identifier integer ranging from 1 to 953.

## Milestone 3
### Evaluating the Model In Terms of Training and Testing Error
Referring back to the notebook cell results, more specifically cell 58, we notice that the training and testing error are roughly of the same magnitude, which does not suggest overfitting but a case of underfitting. In this case, we may need to select a better model in the future or perform rigorous tuning in order to bring the MSE closer to 0.

### Where does your model fit in the fitting graph?
Our top 3 features correlated with `in_total_playlists` were `streams`, `released_year`, and `in_apple_charts`. When we used all 3 of these input features for the linear regression, we did not obtain a clean straight line as we would have if we used any 1 of these features by itself as the input feature. The graphs of `streams` and `released_year` are shown below:

![streams and in_total_playlists](/assets/M3%20graph%201.png)
![released_year and in_total_playlists](/assets/M3%20graph%202.png)
<!-- ![in_apple_charts and in_total_playlists](/assets/M3%20graph%203.png) -->

In order to roughly see how the individual input features contributed to the output by themselves while holding the other values constant, we relied on partial dependence graphs shown below:

![streams and in_total_playlists](/assets/M3%20graph%204.png)
![released_year and in_total_playlists](/assets/M3%20graph%205.png)

By observation, the regression seems to have minimal signs of overfitting, but a polynomial regression may be more optimal as a curve could help us reduce error especially in features such `in_apple_charts`.

### What are the next 2 models you are thinking of and why?
Building on the linear regression model, our exploration into more sophisticated modeling techniques leads us to consider Polynomial Regression and SVR, each with unique attributes that make them suitable for our dataset's specific challenges.

After working with linear regression, we are thinking of a few models such as Polynomical regression to account for the curved path that the data follows. In our linear model, we noticed the residuals scattered and noted that a polynomial regression would be a potentially better fit to try out. By incorpororating polynomial variables, this allows us to model a wide range of graphs. The flexibility to model for these non-linear trends is crucial in accurately representing the complexity of our dataset, which includes diverse metrics from platforms like Shazam, Spotify, and Apple Music. However, the challenge with Polynomial Regression lies in selecting the right degree for the polynomial. Setting too high of degree can lead to overfitting, where the model becomes overly complex and starts capturing noise, while too low of a degree might not capture the data's traits properly.

SVR is another model we're considering, particularly due to the complexity and potential high dimensionality of our dataset. SVR stands out with its use of kernel functions, which allow us to model complex, non-linear relationships without explicitly increasing the model's complexity through polynomial expansion. This aspect of SVR is particularly beneficial for our data, as it involves various dimensions and features from different music streaming platforms. The kernel trick in SVR enables us to explore these relationships in a more clear manner, potentially leading to more accurate predictions across different platforms. Moreover, SVR's ability to handle high-dimensional spaces is advantageous given the diverse range of features in our data, from streaming counts to musical characteristics. This makes SVR a great choice for achieving a balance between model accuracy and complexity, ensuring that we're not just fitting to random variables or missing out on critical patterns.

### What is the conclusion of your 1st model? What can be done to possibly improve it?
The conclusion that we can come to from our 1st model is that linear regression is too simple of a model for our data. Our MSE for both training and test sets were very high meaning that there was a huge difference between the data and the predicted values. As seen by the plots, our data does not really follow a linear trend so it would make sense that our MSE is very high. Our data is too complex to be fitted using linear regression, so to improve it we can use more complex models like polynomial regression. With polynomial regression, we can more accurately fit our data to account for curved patterns. However, using too low of a degree might still lead to underfitting, and choosing too high of a degree can lead to overfitting.

