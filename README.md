# CSE151A - Machine Learning Final Project

## Project Link
https://colab.research.google.com/github/varunraisinghal/CSE151A-MachineLearningFinalProject/blob/main/CSE151A_Machine_Learning.ipynb

## Navigation Links
[Milestone 2 - Data Preprocessing](#milestone-2---data-preprocessing)\
[Milestone 3 - Model 1](#milestone-3---model-1)\
[Milestone 4 - Model 2](#milestone-4---model-2)\
[Milestone 5 - Model 3 and Final](#milestone-5---model-3-and-final)

## Introduction
### Proposed Main Dataset: 
https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023

### Dataset Description:
The dataset is titled: “Most Streamed Spotify Songs 2023" and offers a comprehensive list of ~1000 of the most popular songs on Spotify for 2023. It includes a wide range of data points that go beyond typical song datasets, providing a deep dive into the attributes and popularity of each track across various music platforms. 

### Key features of this dataset include:
1. Track Name: The title of the song.
2. Artist(s) Name: The name(s) of the artist(s) who performed the song.
3. Artist Count: The number of artists contributing to the song.
4. Release Date: The year, month, and day when the song was released.
5. BPM: Beats per minute, a measure of song tempo
6. Key: Key of the song
7. Mode: Mode of the song (major or minor)
8. Spotify Metrics: Includes the number of Spotify playlists the song is in, its presence and rank on Spotify charts, and total streams on Spotify.
9. Apple Music Metrics: Number of Apple Music playlists featuring the song and its rank on Apple Music charts.
10. Deezer Metrics: Similar data for Deezer, including playlist inclusion and chart ranking.
11. Shazam Charts: The song's presence and rank on Shazam charts.
12. Audio Features: Various characteristics of the song such as BPM (beats per minute), key, mode, danceability, valence, energy, acousticness, instrumentalness, liveness, and speechiness, usually given in percentages.

### Abstract:
This project introduces an innovative approach to forecast the number of playlist a song is in using the Spotify dataset, an extensive repository of music and associated features. Our goal is to predict the inclusion of songs in playlists across major music streaming platforms, such as Spotify, Apple Music, and Deezer, using a comprehensive dataset of the most streamed Spotify songs of 2023.  Utilizing the machine learning techniques we use in this course, we can analyze the dataset to identify patterns and correlations between a song's features and its popularity. This involves an examination of the various features contained within the Spotify dataset, such as musical composition, lyrical content, and genre-specific elements. Our analysis utilizes the rich data provided by the Spotify dataset, which encompasses a wide array of audio features and streaming metrics. We aim to see the underlying trends driving the music industry's hit productions and identify the key factors contributing to a song's success. This approach seeks to bridge the gap between the quantitative data of song popularity and the qualitative aspects of musical creativity. Ultimately, our research intends to offer valuable insights into the preferences of music, supporting a more data-informed strategy in hit song production.

### Why This Project? 
We chose this project because we thought it would be interesting to see how music producers produce a hit song. We wanted to see if there was an underlying formula, or if these producers were just randomly producing great hits. Historically, predicting a hit song was often left to the intuition of music executives and producers. However, by applying data science to music analytics, we can identify quantifiable patterns and trends that contribute to a song's popularity. This makes the field of producing music level for those wanting to make a hit song just like the top charts. 

### Why is it cool?
The coolest aspect of this project lies in its capacity to convert the abstract qualities of music—such as melody, rhythm, lyrics, and genre—into quantifiable data that can be analyzed and predicted. Imagine unveiling a formula that could predict the next chart-topper, leveraging the complex interplay of beats per minute, lyrical sentiment, and genre trends. This project is thrilling because it offers a glimpse into the future of music production, where data and creativity converge to create hits. It challenges the traditional notion that hit songs are born solely from creative genius, proposing instead that there's a patterned science behind music that captivates the masses. Our fascination with understanding what makes a song go viral or become a one-hit wonder propels this project into an exciting exploration of music analytics.

### Impact of a Good Predictive Model
The potential impact of an accurate predictive model for song popularity is vast, extending beyond the music industry to influence how content is curated, marketed, and consumed across global digital platforms. For artists and producers, such a model could offer a blueprint for success, enabling them to craft songs with elements that are more likely to appeal to their target audiences. Record labels could optimize their investment strategies, focusing on artists and projects with the highest potential for commercial success. Streaming platforms, on the other hand, could enhance their recommendation algorithms, improving user experience by connecting listeners with songs that are more aligned with their tastes and preferences. 
Moreover, the broader cultural impact of this project could be significant, fostering a greater diversity of music that has the potential to become popular. By understanding the dynamics of hit song production, the industry can move away from a one-size-fits-all approach, encouraging a wider range of artistic expressions to flourish. In essence, a good predictive model does not just predict hits; it could potentially redefine what a hit can be, contributing to a richer, more diverse musical landscape.


## Milestone 2 - Data Preprocessing
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

## Milestone 3 - Model 1
### Evaluating the Model In Terms of Training and Testing Error
Referring back to the notebook cell results, more specifically cell 58, we notice that the training and testing error are roughly of the same magnitude, which does not suggest overfitting but a case of underfitting. In this case, we may need to select a better model in the future or perform rigorous tuning in order to bring the MSE closer to 0.

### Where does your model fit in the fitting graph?
Our top 3 features correlated with `in_total_playlists` were `streams`, `released_year`, and `in_apple_charts`. When we used all 3 of these input features for the linear regression, we did not obtain a clean straight line (shown in red as a scatterplot for visibility) as we would have if we used any 1 of these features by itself as the input feature. The graphs of `streams` and `released_year` are shown below:

![streams and in_total_playlists](/assets/M3%20graph%201%20scatter.png)
![released_year and in_total_playlists](/assets/M3%20graph%202%20scatter.png)
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

## Milestone 4 - Model 2
### Evaluate your data, labels and loss function. Were they sufficient or did you have have to change them.
In the development of our second model, we implemented minor yet crucial adjustments in our data processing and loss function usage to enhance the model's performance. Initially, we shifted our data standardization approach from a basic linear approximation to min-max normalization. This transition was pivotal for preparing our dataset, comprised solely of numerical features (excluding the `in_total_playlists` column, which was designated as the target variable y), for neural network algorithms. Neural networks are notably sensitive to the scale of input data, and by normalizing all numerical features to a common range between 0 and 1 using the MinMaxScaler from the scikit-learn library, we significantly improved the model's convergence speed and overall efficiency in identifying complex patterns within the data. The decision to maintain a numerical feature set was influenced by the relevance of these features to our predictive goals, allowing us to concentrate on refining other aspects of the model without altering the data structure significantly. As for the loss function, we chose the Mean Squared Error (MSE), initially used as an evaluation metric in linear models, as our primary loss function for the neural network. This decision was based on MSE's proven effectiveness in quantifying the difference between the model's predictions and the actual values, thereby supporting our aim to enhance model accuracy and performance in the transition to more complex neural network algorithms.

### Evaluate your model compare training vs test error
After compiling our neural network model, it was trained using the model.fit method with the input features X_train and target variable y_train. To monitor the model's performance and mitigate overfitting, a validation set comprising 10% of the training data was established through the validation_split parameter. Additionally, an early stopping callback was utilized to halt training if no improvement in validation loss was observed over a series of epochs, enhancing the training process's efficiency and preventing overfitting. The training was conducted silently with the verbose parameter set to 0, meaning that updates on the training progress were not displayed. The model's accuracy and generalization capability were assessed by calculating the mean squared error (MSE) for both training and testing datasets, resulting in a Training MSE of roughly 16,000,000 and a Test MSE of 17,500,000. These metrics indicate the model's performance, with the difference between the training and test MSE highlighting the model's ability to generalize to new, unseen data.

### Where does your model fit in the fitting graph, how does it compare to your first model?
With the fitting graph, we first looked at the model's `in_total_playlists` prediction with respect to all of the numeric features. We compare the predicted `in_total_playlists` values with the input feature `streams` below:\
\
The first image below is the graph from model 1 (linear regression).
![streams and in_total_playlists](/assets/M4%20graph%201.png)\
\
The second image below is the graph from model 2 (neural network).
![streams and in_total_playlists](/assets/M4%20graph%202.png)\
\
(For clarity, both models were run using all numeric features). By observation, we can see that the model 1 linear regression's predictions were restricted by the linear nature of this model, hence why we see a slight red line forming that cannot cover many of the blue dots (actual values) below it. By contrast, the neural network is able to predict values more accurately, hence why we see more red dots (predictions) covering blue dots (actual values).\
\
To see each feature's independent contribution to the output, we relied once more on partial dependence plots shown below:\
![streams and in_total_playlists](/assets/M4%20graph%203.png)\
\
In comparison to the model 1's partial dependence plots, we can see that model 2 has generated a subtle curve for each of the input features. Though it is hard to notice, this curve likely fits the data better, which has led to model 2 having a lower MSE than model 1 on new data. We can see this with `streams` below:\
\
The first image below is the graph from model 1 (linear regression).\
![streams and in_total_playlists](/assets/M4%20graph%204.png)\
\
The second image below is the graph from model 2 (neural network).\
![streams and in_total_playlists](/assets/M4%20graph%205.png)

### Did you perform hyper parameter tuning? K-fold Cross validation? Feature expansion? What were the results?
In our most recent development phase, contrary to our initial approach, we actively engaged in implementing advanced techniques such as hyperparameter tuning and k-fold cross-validation, alongside integrating callbacks like early stopping and model checkpointing to refine our neural network's training process. We established a static learning rate of 0.001 and utilized a 10-fold cross-validation scheme, dividing the data into 10 folds to ensure a robust evaluation of the model's generalizability across different data subsets. With a batch size of 100 and a training epoch limit set to 1000, we optimized our training regimen to balance between computational efficiency and model performance. Our methodology involved iterating through each fold defined by a KFold object with 10 splits, training the model on each split, and calculating the Mean Squared Error (MSE) for both training and test sets. This comprehensive evaluation strategy allowed us to assess the model's performance consistently across different partitions of the dataset, enhancing the reliability of our findings. By incorporating an early stopping callback, we mitigated the risk of overfitting by halting the training process if the validation loss showed no improvement over a designated number of epochs. Simultaneously, the model checkpointing ensured the preservation of the best model state throughout the training process.

### What is the plan for the next model you are thinking of and why?
Our next step in improving our current numerical-based neural network involves a focused application of hyperparameter tuning, gradient descent, and k-fold cross-validation. The objective is to fine-tune the model's parameters using hyperparameter tuning, which allows us to systematically search for the most effective configuration. This method is crucial for enhancing model performance and efficiency. Gradient descent will be used to optimize the model further by adjusting parameters to minimize the cost function, a key step in improving the accuracy of predictions. We also plan to implement k-fold cross-validation, a robust method for assessing the model's generalizability. This involves dividing the dataset into k smaller sets to validate the model on one set while training on the others. This process helps prevent overfitting and ensures the model performs well on unseen data. This approach marks a significant improvement over our basic neural network by employing sophisticated techniques aimed at refining the model's accuracy and generalizability. Our goal is to not only enhance the performance of our neural network but also to ensure it remains adaptable and reliable across different datasets.

### What is the conclusion of your 2nd model? What can be done to possibly improve it? How did it perform to your first and why?
Our second model was a step up from the first model in terms of complexity and results. When using all numeric input features, model 1 had a testing MSE >36,000,000. Model 2 in comparison had a testing MSE of ~20,000,000 (~24,000,000 for kfold cross validation). Model 2, being a neural network, can account for more complex and non-linear relationships that linear regressions struggle with. Model 1's linear regression, as shown in previous figures, predicted poorly especially with clusters of points that are difficult to predict with just a line. Model 2 however was able to generate curves to better account for this complexity. For improvement, we can perform more hyperparameter tuning with the number of layers, number of nodes per layer, the optimizer, the activation function, etc. Though this would take a lot of time, this would ensure that the most optimal parameters are chosen to minimize the MSE and get more accurate predictions.

## FIGURES/ PICTORIAL REPRESENTATIONS 
The Python notebook on GitHub contains various images depicting data visualizations and model analyses, such as histograms, scatter plots, and heatmaps, highlighting key aspects of the project like data distribution, feature correlations, and model performance. 
The data exploration involved examining track features from Spotify, identifying patterns, and handling missing or erroneous values. Preprocessing techniques included normalizing data, encoding categorical variables, and feature selection to prepare for modeling. Models were developed to predict track popularity, with each model's choice and parameters based on initial findings and performance metrics.

### Figures in Exploratory Data Analysis:
Performing data analysis on large datasets is key to predicting, and understanding deeply on what are its underlying features and what can be extracted and used from it. These visual tools are vital for making informed decisions on data preprocessing and model selection strategies.

This visual subplots of different features in the dataset gives an insight into the range, distribution type which helps us standardize data and augment it potentially in the future.

This heatmap gives a very detailed correlation between features. By observing how these features correlate with one another—for instance, whether more danceable tracks tend to be more energetic—you can make more informed decisions on feature selection and engineering for predictive modeling. 

### Figures for Model 1 - Linear Regression Model:

The image shows a series of partial dependence plots from a predictive model, testing the relationship between several features and the target variable. Observing the 'Fit' line and the distribution of 'Actual' data points, we can assess whether and how well the model captures the trends in the data for each feature individually. 

### Figures for Model 2 (DFFNN with RELU activation function):

### Figures for Model 3:

## METHODS

### Model 1 (Linear Regression):
For Model 1, we use a Linear Regression model with the Top 3 Correlating Features, we'll delve into a more detailed explanation of the methods section. This section will be structured to outline the methodology, focusing on each step from data exploration through to the modeling process.

#### Data Exploration
The initial step involved a comprehensive analysis of the Spotify dataset to gain insights into the characteristics and relationships within the data. This phase focused on identifying patterns, outliers, and the underlying distribution of data points. Special attention was given to the correlation between various features and the target variable, in_total_playlists, to pinpoint the most influential predictors. The top 3 features with the highest correlation to the target variable were selected for further analysis. This selection was based purely on statistical measures of correlation, ensuring an objective approach to feature selection.

#### Preprocessing
The preprocessing stage involved preparing the dataset for modeling. This included the following key steps:
Data Splitting: The dataset was split into training and testing sets, to an 80/20 split. This division ensured a robust evaluation of the model's performance, providing a separate dataset for training and another for validation to test the model's prediction on unseen data.
Feature Selection: Based on the correlation analysis during the data exploration phase, the top 3 correlating features were isolated from the dataset. This step refined the input variables to those most relevant for predicting the target variable, streamlining the model's focus and potentially improving its performance.

#### Model Details:
The modeling process was centered around Linear Regression, a fundamental technique for predicting continuous variables. This choice was made by the model's simplicity, and suitability for establishing linear relationships between the selected features and the target variable.

#### Model Training: 
The Linear Regression model was trained using the selected features from the preprocessing stage. The training process involved adjusting the model's parameters to minimize the difference between the predicted and actual values of the target variable within the training dataset. This iterative process aimed to find the best-fit line that could generalize well to new, unseen data.

#### Model Evaluation: 
To assess the model's performance, the Mean Squared Error (MSE) metric was employed. MSE provided a quantitative measure of the model's accuracy by calculating the average squared difference between the predicted and actual values. 

#### Visualization: 
Most of the visualizations were provided in the previous section but visualizing the model's predictions against the actual values gave us a more intuitive understanding of the model’s power. The discrepancies and high value of error made us realize that a simple linear regression model wouldn’t be fully sufficient. Through scatter plots depicting the relationship we could conclude what features were best of use to continue using and gave us a direction to move forward with the project.

### Model 2 (DFFNN with RELU activation function):
Model 2 uses a Deep Feedforward Neural Network (DFFNN) with the ReLU activation function, leveraging all numerical features from the Spotify dataset. This model represents a more complex approach than linear regression, aiming to capture nonlinear relationships within the data through a multi-layer architecture.
#### Architecture Summary:
Input Layer: 17 neurons (for each numerical feature)
Hidden Layer 1: 17 neurons, ReLU activation
Hidden Layer 2: 34 neurons, ReLU activation
Hidden Layer 3: 34 neurons, ReLU activation
Hidden Layer 4: 17 neurons, ReLU activation
Output Layer: 1 neuron, ReLU activation

#### Data Exploration and Preprocessing
Before the neural network model was constructed, we preprocessed all numerical features within the Spotify dataset. This preprocessing step involved scaling the data using MinMaxScaler to normalize feature values between 0 and 1, a crucial step for neural network models to ensure that no particular feature dominates the learning process due to its scale. The dataset was then split into training and test sets with an 80/20 ratio, maintaining a balance between training the model and evaluating its performance on unseen data.


#### Training Strategy
The training strategy incorporated several key components to optimize performance and prevent overfitting:

#### Optimizer: 
Adam optimizer was selected with a static learning rate of 0.001. This optimizer is widely used for its adaptiveness in handling sparse gradients on noisy problems.

#### Loss function: 
Mean Squared Error (MSE) was used as the loss function, aligning with the model's goal to minimize the difference between predicted and actual playlist counts. This is what we kept the same from our model 1 as well.
Early Stopping: Implemented to halt training if the validation loss did not improve for 100 epochs, helping prevent overfitting by restoring model weights.

#### Evaluation and Analysis
The model was evaluated based on its MSE on both training and test datasets. Notably, the training MSE was significantly lower than the test MSE, indicating the model's ability to learn from the training data but also suggesting a potential overfitting to the training set or underperformance on the test set. Hence, we perform K-Fold Cross Validation on this and we see results for it after. 

Visualizations played a crucial role in interpreting the model's performance. Scatter plots comparing actual vs. predicted values for both training and test data provided insights into the model's accuracy and areas of improvement. Partial Dependence Plots (PDPs) were used to understand the effect of each feature on the prediction outcome, for a deeper look into the model's internal decision-making process.



## Model 3 (Extra Trees Regressor):
#### Data Preparation
The dataset includes features like artist count, release year, streams, and various musical attributes (e.g., bpm, danceability).Excluded columns related to playlist inclusions on specific platforms and non-numerical features (e.g., track name, artist names) to focus on numerical predictors.The target variable is in_total_playlists, representing the total playlist inclusions across platforms.
Preprocessing
Numerical features were standardized using StandardScaler to ensure model input variables have a mean of 0 and a standard deviation of 1, improving model performance.

#### Model Training and Evaluation
#### Model Used: 
Extra Trees Regressor, trained on the preprocessed numerical features.
#### Fitting the Model: 
The model was trained using the entire set of standardized numerical features from the training dataset.
#### Performance Metrics:
#### Mean Squared Error (MSE): 
10,363,528.63, indicating the average squared difference between the estimated values and the actual value of the total playlist inclusions.
R-squared (R²).







## Milestone 5 - Model 3 and Final
TBA
