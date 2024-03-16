# CSE151A - Machine Learning Final Project

## Project Link

<https://colab.research.google.com/github/varunraisinghal/CSE151A-MachineLearningFinalProject/blob/main/CSE151A_Machine_Learning.ipynb>

## Navigation Links

[Milestone 2: Data Exploration & Initial Preprocessing](#milestone-2)\
[Milestone 3: Model 1](#milestone-3)\
[Milestone 4: Model 2](#milestone-4)\
[Milestone 5: Model 3 and Final](#milestone-5)

<!-- ## Milestone 2: Data Exploration & Initial Preprocessing -->
## Milestone 2

### Tasks Already Done

1. First, we loaded the CSV file and converted it to a pandas Dataframe for ease of manipulation.
2. Next, after noticing missing values within the `in_shazam_charts` and `key` feature columns, we decided to do two things. First, we replaced the `in_shazam_charts` NA values with `0` since if the value for `in_shazam_charts` is not applicable, then it must not be in the Shazam charts, so it would in turn be false (aka `0`). Second, we decided, for now, to completely drop the `key` column since there were 95 missing values which will likely detract from our model without imputation. We will also drop the `mode` column, since it is related to the key and not necessary for objective.
3. Then, we check the datatypes of our feature columns. `in_shazam_charts` is `float64` because of the NA values. Since we have replaced those, we can cast this feature column to `int64`. Another feature column, `streams`, is an `object` when it should be a numeric type. This likely indicates one or more of the entries are faulty, so we can remove that entry and then typecast the feature column as `int64`.
4. We used the `.describe()` method to provide a summary of the numerical columns in the dataset. We wanted to do this to get a good understanding of the basic statistical details like mean, standard deviation, minimum, and maximum values of each numeric feature so that we can get a good sense of our data.
5. We also created histograms for all the various features to help understand the distribution of the characteristics of the songs. Heatmap was created as well as a pairplot in order to identify trends and analyze how different features correlate to each other.

### Potential Tasks To Be Done

1. Depending on the determined necessity, we may impute the `key` column missing values if the inclusion of the `key` column improves our model's performance.
2. After determining the most relevent features to our model, which as of now are the characteristic percentages of the songs \[`danceability_%`, `valence_%`, `energy_%`, `acousticness_%`, `instrumentalness_%`, `liveness_%`, `speechiness_%`\], we will isolate those features from our dataset.
3. Then, we will divide our relavent dataset into separate training and testing datsets using a 80/20 random split. We may modify the proportions if we recognize signs of over or under fitting.
4. Since currently all our features are percentages that range from 0 to 100, we will not have to standardize our dataset. However, if we decide to include other features, we may have to standardize our values.
5. Since every song name is unique and non-numeric, we may have to assign each song a unique identifier integer ranging from 1 to 953.

<!-- ## Milestone 3: Model 1 - Linear Regression Model -->
## Milestone 3

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

<!-- ## Milestone 4: Model 2 - DFFNN with RELU -->
## Milestone 4

### Evaluate your data, labels and loss function. Were they sufficient or did you have have to change them?

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

<!-- ## Milestone 5: Final Submission -->
## Milestone 5

## Introduction

### Proposed Main Dataset

<https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023>

### Dataset Description

The dataset is titled: “Most Streamed Spotify Songs 2023" and offers a comprehensive list of ~1000 of the most popular songs on Spotify for 2023. It includes a wide range of data points that go beyond typical song datasets, providing a deep dive into the attributes and popularity of each track across various music platforms.

### Key features of this dataset include

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

### Abstract

This project introduces an innovative approach to forecast the number of playlist a song is in using the Spotify dataset, an extensive repository of music and associated features. Our goal is to predict the inclusion of songs in playlists across major music streaming platforms, such as Spotify, Apple Music, and Deezer, using a comprehensive dataset of the most streamed Spotify songs of 2023.  Utilizing the machine learning techniques we use in this course, we can analyze the dataset to identify patterns and correlations between a song's features and its popularity. This involves an examination of the various features contained within the Spotify dataset, such as musical composition, lyrical content, and genre-specific elements. Our analysis utilizes the rich data provided by the Spotify dataset, which encompasses a wide array of audio features and streaming metrics. We aim to see the underlying trends driving the music industry's hit productions and identify the key factors contributing to a song's success. This approach seeks to bridge the gap between the quantitative data of song popularity and the qualitative aspects of musical creativity. Ultimately, our research intends to offer valuable insights into the preferences of music, supporting a more data-informed strategy in hit song production.

### Why This Project?

We chose this project because we thought it would be interesting to see how music producers produce a hit song. We wanted to see if there was an underlying formula, or if these producers were just randomly producing great hits. Historically, predicting a hit song was often left to the intuition of music executives and producers. However, by applying data science to music analytics, we can identify quantifiable patterns and trends that contribute to a song's popularity. This makes the field of producing music level for those wanting to make a hit song just like the top charts.

### Why is it cool?

The coolest aspect of this project lies in its capacity to convert the abstract qualities of music—such as melody, rhythm, lyrics, and genre—into quantifiable data that can be analyzed and predicted. Imagine unveiling a formula that could predict the next chart-topper, leveraging the complex interplay of beats per minute, lyrical sentiment, and genre trends. This project is thrilling because it offers a glimpse into the future of music production, where data and creativity converge to create hits. It challenges the traditional notion that hit songs are born solely from creative genius, proposing instead that there's a patterned science behind music that captivates the masses. Our fascination with understanding what makes a song go viral or become a one-hit wonder propels this project into an exciting exploration of music analytics.

### Impact of a Good Predictive Model

The potential impact of an accurate predictive model for song popularity is vast, extending beyond the music industry to influence how content is curated, marketed, and consumed across global digital platforms. For artists and producers, such a model could offer a blueprint for success, enabling them to craft songs with elements that are more likely to appeal to their target audiences. Record labels could optimize their investment strategies, focusing on artists and projects with the highest potential for commercial success. Streaming platforms, on the other hand, could enhance their recommendation algorithms, improving user experience by connecting listeners with songs that are more aligned with their tastes and preferences.
Moreover, the broader cultural impact of this project could be significant, fostering a greater diversity of music that has the potential to become popular. By understanding the dynamics of hit song production, the industry can move away from a one-size-fits-all approach, encouraging a wider range of artistic expressions to flourish. In essence, a good predictive model does not just predict hits; it could potentially redefine what a hit can be, contributing to a richer, more diverse musical landscape.
<!-- ------------------------------------------------------------------------------------------------------------->
## FIGURES/ PICTORIAL REPRESENTATIONS
The Python notebook on GitHub contains various images depicting data visualizations and model analyses, such as histograms, scatter plots, and heatmaps, highlighting key aspects of the project like data distribution, feature correlations, and model performance. 
The data exploration involved examining track features from Spotify, identifying patterns, and handling missing or erroneous values. Preprocessing techniques included normalizing data, encoding categorical variables, and feature selection to prepare for modeling. Models were developed to predict track popularity, with each model's choice and parameters based on initial findings and performance metrics.

### Figures in Exploratory Data Analysis:
Performing data analysis on large datasets is key to predicting, and understanding deeply on what are its underlying features and what can be extracted and used from it. These visual tools are vital for making informed decisions on data preprocessing and model selection strategies.

This visual histograms of different features in the dataset gives an insight into the range, distribution type which helps us standardize data and augment it potentially in the future.

![Distribution Histograms](/final%20assets/histograms.png)


This heatmap gives a very detailed correlation between features. By observing how these features correlate with one another—for instance, whether more danceable tracks tend to be more energetic—you can make more informed decisions on feature selection and engineering for predictive modeling.

![Correlation Heatmap](/final%20assets/heatmap.png)

The pairplots between the top 3 features and `in_total_playlists` can give us an insight as to how each feature is mathematically related to each other. This can give potential clues as to which models may be suitable for this regression task. For example, `streams` and `in_total_playlists` appear to be linearly correlated, which may signal that a linear regression is a good model to begin with.

![Pairplots](/final%20assets/pairplots.png)

<!-- ------------------------------------------------------------------------------------------------------------->
## Methods

### Data Exploration

The initial step involved a comprehensive analysis of the Spotify dataset to gain insights into the characteristics and relationships within the data. This phase focused on identifying patterns, outliers, and the underlying distribution of data points. Special attention was given to the correlation between various features and the target variable, in_total_playlists, to pinpoint the most influential predictors. The top 3 features with the highest correlation to the target variable were selected for further analysis. This selection was based purely on statistical measures of correlation, ensuring an objective approach to feature selection.


### Preprocessing

Preprocessing techniques included handling missing or erroneous values, normalizing data, encoding categorical variables, and feature selection to prepare for modeling. Models were developed to predict track popularity, with each model's choice and parameters based on initial findings and performance metrics. For each model, the dataset was split into training and testing sets, to an 80/20 split. This division ensured a robust evaluation of the model's performance, providing a separate dataset for training and another for validation to test the model's prediction on unseen data.\
\
Based on the correlation analysis during the data exploration phase, the top 3 correlating features were isolated from the dataset. This step refined the input variables to those most relevant for predicting the target variable, streamlining the model's focus and potentially improving its performance.

![Sample output of DataFrame](/final%20assets/dataframe.png)

### Model 1 - Linear Regression Model

For Model 1, we use a Linear Regression model with the Top 3 Correlating Features. This section will be structured to outline the methodology, focusing on each step of the modeling process.

#### Model Details

The modeling process was centered around linear regression, a fundamental and straightforward technique for predicting continuous variables.

#### Model Training

The Linear Regression model was trained using the selected top 3 features from the preprocessing stage. The training process involved adjusting the model's parameters to minimize the difference between the predicted and actual values of the target variable within the training dataset.

#### Model Evaluation

To assess the model's performance, the Mean Squared Error (MSE) metric was employed, a common loss metric for regressions.

#### Visualizations

We used Scatter plots comparing actual (y) vs. predicted values (yhat) as a basic way to assess accuracy. We also did fitting graphs, with each input feature plotted against the output. Partial Dependence Plots (PDPs) were used to see the effect of each individual feature on the prediction outcome.

#### Implementation 
```
def m1_linear_regression(data, in_features, out_feature):
    X_train, X_test, y_train, y_test = train_test_split(data[in_features], data[out_feature], test_size=0.20, random_state = 42)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    yhat_train = lin_reg.predict(X_train)
    yhat_test = lin_reg.predict(X_test)

    run_analytics(lin_reg, X_train, X_test, y_train, y_test, yhat_train, yhat_test, in_features, out_feature, 1)
    
m1_linear_regression(spotify_data, top_3_features, 'in_total_playlists')
```

### Model 2 - DFFNN with RELU

Model 2 uses a Deep Feedforward Neural Network (DFFNN) with the ReLU activation function, leveraging all numerical features from the Spotify dataset.

#### Architecture Summary

Input Layer: 17 neurons (for each numerical feature)
Hidden Layer 1: 17 neurons, ReLU activation
Hidden Layer 2: 34 neurons, ReLU activation
Hidden Layer 3: 34 neurons, ReLU activation
Hidden Layer 4: 17 neurons, ReLU activation
Output Layer: 1 neuron, ReLU activation

#### Data Exploration and Preprocessing

Before the neural network model was constructed, we preprocessed all numerical features within the Spotify dataset. This preprocessing step involved scaling the data using MinMaxScaler to normalize feature values between 0 and 1. The dataset was then split into training and test sets with an 80/20 ratio.

#### Training Strategy

The training strategy incorporated several key components to optimize performance and prevent overfitting: 

#### Optimizer

Adam optimizer was selected with a static learning rate of 0.001.

#### Loss function

Mean Squared Error (MSE) was used as the loss function, aligning with the model's goal to minimize the difference between predicted and actual playlist counts. This is what we kept the same from our model 1 as well.\
\
We also used early stopping to halt training if the validation loss did not improve for 100 epochs.

#### Evaluation and Analysis

The model was evaluated based on its MSE on both training and test datasets. We also performed K-Fold Cross Validation on this model to get comparison results from randomly chosen folds of the dataset.

#### Visualizations

Scatter plots comparing actual (y) vs. predicted values (yhat) for both training and test data provided insights into the model's accuracy and areas of improvement. We also did fitting graphs, with each input feature plotted against the output. Partial Dependence Plots (PDPs) were used to see the effect of each individual feature on the prediction outcome.

#### Implementation 
```
def m2_neural_network(data, in_features, out_feature):
    X = MinMaxScaler().fit_transform(data[in_features])
    y = data[out_feature]

    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(X, columns = in_features), y, test_size=0.20, random_state = 42)

    activationMethod = "relu"
    metrics = ["mean_squared_error"]
    loss_type = "mean_squared_error"
    lr_static = 0.001
    batch_size=100
    epochs=1000

    model = keras.Sequential()
    model.add(keras.Input(shape=np.shape(in_features)))
    model.add(layers.Dense(17, activation=activationMethod, name="layer1"))
    model.add(layers.Dense(34, activation=activationMethod, name="layer2"))
    model.add(layers.Dense(34, activation=activationMethod, name="layer3"))
    model.add(layers.Dense(17, activation=activationMethod, name="layer4"))
    model.add(layers.Dense(1, activation=activationMethod, name="output"))

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.95) #Found on the keras.io/api/optimizers page as an example of dynamic learning rate

    opt = keras.optimizers.Adam(
        learning_rate=lr_static
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=100,
        verbose=1,
        mode='min',
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=100
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="./modelCheckpoints",
        monitor='val_loss',
        verbose = 0,
        save_best_only = False,
        save_weights_only = False,
        mode = 'auto',
        save_freq = 'epoch',
        options = None,
        initial_value_threshold = None
    )

    model.compile(optimizer=opt, loss=loss_type, metrics=metrics)

    kr_model = KerasRegressor(model, loss = loss_type, metrics = metrics, verbose = 0)

    history = kr_model.fit(X_train.astype('float'), y_train, validation_split=0.1, callbacks = [early_stopping], batch_size=batch_size, epochs=epochs, verbose=0)

    yhat_train = kr_model.predict(X_train, verbose=0)
    yhat_test = kr_model.predict(X_test, verbose=0)

    run_analytics(kr_model, X_train, X_test, y_train, y_test, yhat_train, yhat_test, in_features, out_feature, 2)

    y = np.array(y)
    MSE_train = []
    MSE_test = []

    R2_train = []
    R2_test = []

    num_splits = 10
    kf = KFold(n_splits=num_splits)

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        history = model.fit(X[train_index].astype('float'), y[train_index], validation_split=0.1, callbacks = [early_stopping], batch_size=batch_size, epochs=epochs, verbose=0)
    
        fold_pred_train = model.predict(X[train_index])
        fold_pred_test = model.predict(X[test_index])

        mse_train = mean_squared_error(y[train_index], fold_pred_train)
        mse_test = mean_squared_error(y[test_index], fold_pred_test)
        r2_train = r2_score(y[train_index], fold_pred_train)
        r2_test = r2_score(y[test_index], fold_pred_test)
        print("Fold " + str(i) + " Training MSE: " + str(mse_train))
        print("Fold " + str(i) + " Training R2: " + str(r2_train))
        print("Fold " + str(i) + " Testing MSE: " + str(mse_test))
        print("Fold " + str(i) + " Testing R2: " + str(r2_test) + '\n')

        MSE_train.append(mse_train)   
        MSE_test.append(mse_test)
        R2_train.append(r2_train)   
        R2_test.append(r2_test)

    print("\nOverall Training MSE: " + str(np.array(MSE_train).mean()))
    print("Overall Training R2: " + str(np.array(R2_train).mean()))
    print("\nOverall Testing MSE: " + str(np.array(MSE_test).mean()))
    print("Overall Testing R2: " + str(np.array(R2_test).mean()))

m2_neural_network(spotify_data, all_features, 'in_total_playlists')
```

### Model 3 - Random Forest

#### Data Preparation

The dataset includes features like artist count, release year, streams, and various musical attributes (e.g., bpm, danceability).Excluded columns related to playlist inclusions on specific platforms and non-numerical features (e.g., track name, artist names) to focus on numerical predictors.The target variable is in_total_playlists, representing the total playlist inclusions across platforms.
Preprocessing
Numerical features were standardized using StandardScaler to ensure model input variables have a mean of 0 and a standard deviation of 1, improving model performance.

#### Model Training and Evaluation

#### Model Used

Random Forest, trained on the preprocessed numerical features.

#### Fitting the Model

The model was trained using the entire set of standardized numerical features from the training dataset.

#### Performance Metrics

#### Mean Squared Error (MSE)

10,363,528.63, indicating the average squared difference between the estimated values and the actual value of the total playlist inclusions.
R-squared (R²).

#### Implementation 

```
def m3_random_forest(data, in_features, out_feature):
    X_train, X_test, y_train, y_test = train_test_split(data[in_features], data[out_feature], test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), in_features),
        ],
        remainder='passthrough'  # columns not specified are left untouched
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])

    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [10, 20, 30, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__max_features': [1.0, 'sqrt', 'log2']
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    yhat_train = best_model.predict(X_train)
    yhat_test = best_model.predict(X_test)

    print(f'\nBest Hyperparameters: {grid_search.best_params_}')

    run_analytics(best_model, X_train, X_test, y_train, y_test, yhat_train, yhat_test, in_features, out_feature, 3)

m3_random_forest(spotify_data, all_features, 'in_total_playlists')
```

### Model 4 - Extra Trees Regressor

We chose to implement an Extra Trees Regressor because 

#### Implementation
```
def m4_extra_trees(data, in_features, out_feature):
    # train test split with 80% train 20% test using input features
    X_train, X_test, y_train, y_test = train_test_split(data[in_features], data[out_feature], test_size=0.20, random_state = 42)

    # create and train the model
    model = ExtraTreesRegressor(random_state=21)
    model.fit(X_train, y_train)

    # get yhat for training and testing data
    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    # call helper functions
    run_analytics(model, X_train, X_test, y_train, y_test, yhat_train, yhat_test, in_features, out_feature, 4)
    
m4_extra_trees(spotify_data, all_features, 'in_total_playlists')
```
<!-- ------------------------------------------------------------------------------------------------------------->
## Results

### Model 1 - Linear Regression Model

Observing the `y` and `yhat` output comparison plots, we see that the linear regression hovers around 0 when plotting the `yhat - y` after predicting the output based on the top 3 correlated features.\
\
![Model 1 Output Comparison Plots](/final%20assets/m1%20ocp%20test.png)

We can see in the fitting graph the model's predictions mostly cover the actual values, except in the case of `streams`, where it is restricted to a more linear shape.\
\
![Model 1 Fitting Plots](/final%20assets/m1%20fp%20test.png)

The partial dependence plots show the output is strongly correlated with `streams`, and seems to be more weakly correlated with `in_apple_charts` since the line misses the cluster.\
\
![Model 1 Partial Dependence Plots](/final%20assets/m1%20pdp%20test.png)

The testing MSE is ~26,000,000 (number is high because `in_total_playlists` has large values), and the R2 score is ~0.61. These will serve as a baseline to compare with other models.

### Model 2 - DFFNN with RELU

Observing the `y` and `yhat` output comparison plots, we see that the neural network is more closely clustered around 0 when plotting the `yhat - y` after predicting the output based on all numeric features.
\
\
![Model 2 Output Comparison Plots](/final%20assets/m2%20ocp%20test.png)

We can see in the fitting graph the model's predictions mostly cover the actual values, with the `yhat` values being less bound to a particular shape.\
\
![Model 2 Fitting Plots](/final%20assets/m2%20fp%20test.png)

The partial dependence plots shows slight but noticable curves in an attempt to better fit each feature to the output feature.\
\
![Model 2 Partial Dependence Plots](/final%20assets/m2%20pdp%20test.png)

The testing MSE is ~16,000,000, and the R2 score is ~0.76, an improvement over model 1. After 10-fold cross validation, the testing MSE average is ~13,000,000 and the R2 score is ~0.76, indicating minimal signs of overfitting.

### Model 3 - Random Forest

Observing the `y` and `yhat` output comparison plots, we see with the the `yhat - y` that the random forest clusters around 0 more closely than the previous 2 models after predicting the output based on all numeric features.\
\
![Model 3 Output Comparison Plots](/final%20assets/m3%20ocp%20test.png)

We can see in the fitting graph the model's predictions mostly cover the actual values, with the `yhat` values being less bound to a particular shape just like model 2.\
\
![Model 3 Fitting Plots](/final%20assets/m3%20fp%20test.png)

The partial dependence plots shows more aggressively fit lines especially in `streams` in an attempt to strongly fit each feature to the output feature.\
\
![Model 3 Partial Dependence Plots](/final%20assets/m3%20pdp%20test.png)

The testing MSE is ~12,000,000, and the R2 score is ~0.83, a strong improvement over model 1 and 2.

<!-- ------------------------------------------------------------------------------------------------------------->
## Discussion

### Data exploration 

After settling on a dataset to work with, the data exploration involved inspecting the various metrics and variables that went into making a song popular. We looked for key columns that we might think had the most impact on making a song popular. Some observations we had about the data were that there were some columns with missing values. For those columns, we planned on just removing them in the preprocessing step. As for columns with the wrong type, we typecasted columns like ‘streams’ from ‘object/float64’ to ‘int64’ so that we could actually do work on that data.

### Preprocessing  

If the value for in_shazam_charts is not applicable, then it must not be in the Shazam charts, so it would in turn be false (aka 0).  since there were 95 missing values which will likely detract from our model without imputation. since it is related to the key and not necessary for objective. We wanted to do this to get a good understanding of the basic statistical details like mean, standard deviation, minimum, and maximum values of each numeric feature so that we can get a good sense of our data. histogram to help understand the distribution of the characteristics of the songs. Heatmap in order to identify trends and analyze how different features correlate to each other. 

### Model 1 - Linear Regression Model

We chose a linear regression for our first model because of the model's simplicity and suitability for establishing basic linear relationships between the selected features and the target variable. Upon training our model using the top 3 input features correlated to the output feature, predicting the yhat, and attempting to plot, we discovered that we did not fully understand what our independent variable would be on a plot. Since we used multiple input features, when we plotted each yhat with each input feature, we did not get straight lines as expected from a regression. Thus, we relied on partial dependence plots, which keeps all but 1 input feature constant, allowing us to see each input feature's contribution to the output independently. We used MSE as quantitative measure of the model's accuracy by calculating the average squared difference between the predicted and actual values. This error function is used in regressions often, and gave us a baseline value to compare other models to.

### Model 2 - DFFNN with RELU

For the next model, we decided to use a deep feed forward neural network using a relu activation function to improve upon the MSE of the first model. We used a neural network implemented with 4 hidden layers which was a good enough number such that the model would not be too simple like our linear regression model nor would it lead to overfitting because of its increased complexity. Our results for the 2nd model were much more believable than the 1st model because our predicted values started to conform to the general shape of the data for the scatter plots. We saw a slight decrease in MSE suggesting that the model was better than linear regression. This was visible in the partial dependence plots where the curvature in the lines just slightly fit better than a linear regression.

### Model 3 - Random Forest

We used a Random Forest regressor and Extra Trees regressor for our 3rd model with hyperparameter tuning to improve upon the MSE of our first two models. The MSE once again improved as we increased the complexity of our models. Our 2nd model had a pretty good improvement over linear regression. However, for our 3rd model, the addition of hypertuning parameters further improved our MSE. As for the plots, we again plotted each feature on a scatterplot against the data. Like our 2nd model, the scatterplots showed that our predicted values for both training and test sets relatively fit the shape of the actual data. For our partial dependence plots specifically looking at the ‘streams’, which was the highest correlating feature, the line of best fit started to look more complex rather than a straight line or a line with a slight curve. This could have indicated that our models were better generalizing to the data than before.

### Final Thoughts
Although in our final model we arrive at a final trained MSE of ~12,000,000 - this is a much better improvement than in model 1 and model 2. The nature of having such a high MSE is due to the fact that the average mean value of the feature that we are predicting is around 5000 which is reasonable to have an error with a magnitude of that order. It won't be a small increment due to the nature of how big the denomination is. Since we use MSE, we should expect a exponential (square) of the average, resulting in an extremely high value which indicates that our final MSE of ~12,000,000 is not so bad actually.  

<!-- ------------------------------------------------------------------------------------------------------------->
## Conclusion
For this dataset specifically, it was relatively hard to get any one kind of model that fit the data. However, we did continually improve our performance over each milestone by increasing the complexity of the models we used. For the scatterplots, the general shape of our predicted models started to better fit the data over every iteration. However, for the ‘streams’ scatterplot for model 3 our model was wildly off from the data. This could be concerning as ‘streams’ was the feature that most correlated to making a popular song. Other than that, our models improved each time. For the partial dependence plots, there was not much of a visible pattern to the data in the first place. However, we feel that our line of best fit did generalize better to the data over each model. The data overall mostly had clusters of data points, which made it hard for a line of any kind to represent the shape of the data properly. The ‘streams’ feature had the most clear relationship of any feature. The partial dependence plots for ‘streams’ seem to have improved as the models became more complex. For future directions, we could possibly combine multiple models for added complexity, while being cautious of overfitting due to being overly complex. The goal over the entire project was to see if there was anything specific that went into producing a hit song, by capturing a general pattern of any specific feature of a song.

<!-- ------------------------------------------------------------------------------------------------------------->
## Collaboration
Varun Singhal - X Title: 
1. Set up repository, colab notebooks, and etc.. 
2. Helped write abstract document for Milestone 1 and research datasets with group
3. Initial Exploratory Analysis help on Milestone 2
4. What are the next 2 models you are thinking of on Milestone 3
5. What is the conclusion of the models on Milestone 3
6. Completed Milestone 4.1,4.5A,4.5B
7. Final Project #1 - Complete Introduction
8. Final Project #4A - Introduction of Project
9. Final Project #4E - Final Thoughts w/ Jason
          
Eric Tran
1. Researched datasets with group
2. Milestone 3 - What is the conclusion of your 1st model? What can be done to possibly improve it?
3. Completed Final Project #4EF - Discussion and Conclusion

Jason Liang - Developer, Documentation Writer
1. Developed the code for Model 3 and reduced the overall loss 
2. Contributed to documentation writing for Methods and Discussion
3. Assisted in drafting ideas for previous milestones
