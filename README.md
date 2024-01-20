
# **Detect AI Generated Text**

Exploring the landscape of AI-generated texts reveals their diverse applications in Content Generation, Personalized Marketing, Virtual Assistants, and Creative Writing. However, these advancements come with inherent risks such as the spread of misinformation, perpetuation of biases, accountability challenges, and privacy concerns. To navigate these complexities, our project focuses on developing a cutting-edge machine learning algorithm. This algorithm aims to adeptly differentiate between AI-generated and human-generated texts, providing a robust solution to elevate content authenticity and effectively address associated risks.
## Table of Contents

1. [Exploratory Data Analysis](#exploratory-data-analysis)
2. [Prepare Data for Modeling](#prepare-data-for-modeling)
3. [Models](#models)
4. [Evaluation](#evaluation)
5. [Discussion/Conclusion](#discussionconclusion)

##  Import Modules And Data

To start working on the project, follow these steps to import the necessary modules and load the dataset:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# model evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# import other functions we'll need for classification modeling
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from google.colab import drive
drive.mount('/content/drive/')

df = pd.read_csv('/content/drive/Shareddrives/Data Science Project/training_set.csv')
```
``` python
# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns)
print("\nData Types:", df.dtypes)
print("\nDataset Info:")
df.info()
print("\nSummary Statistics:")
df.describe()
```
## **EDA (Exploratory Data Analysis)**

Exploratory Data Analysis (EDA) is a crucial step in understanding the characteristics and patterns within the dataset. Here's an overview of the EDA conducted for this project:

### 1. Target Variable Distribution Analysis
   - Examined the distribution of values (0s and 1s) in the target variable 'ind' to understand the class balance or imbalance within the dataset.
   ```python
   # Target Variable Distribution
target_distribution = df['ind'].value_counts()
print("Target Variable Distribution:")
print(target_distribution)

import matplotlib.pyplot as plt

# Plotting a pie chart for target variable distribution
target_distribution.plot(kind='pie', title='ind Distribution', autopct='%1.1f%%')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.ylabel('')  # Remove the label on y-axis
plt.show()
```

### 2. Top Feature Spread Exploration
   - Investigated the distribution of the top 10 features exhibiting the highest variability or spread across the dataset.
   ```python
   # Distribution of Variables with Maximum Spread
spreads = df.max() - df.min()
sorted_spreads = spreads.sort_values(ascending=False)
feature_spreads = sorted_spreads[sorted_spreads.index.str.startswith('feature_')]

# Selecting the top 10 feature spreads
top_10_feature_spreads = feature_spreads.head(10).reset_index()
top_10_feature_spreads.columns = ['Feature', 'Spread']

# Plotting the top 10 features with maximum features
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[top_10_feature_spreads['Feature']])
plt.title('Box Plot of Top 10 Features')
plt.ylabel('Feature Values')
plt.xticks(rotation=45)
plt.show()

df['feature_497'].describe()
```

### 3. Punctuation Number Distribution
   - Employed a histogram to visually illustrate the contrast in punctuation usage between AI-generated and human-generated text.
   ```python
   from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Define the number of bins for the histogram
n_bins = 30

# Plot the histogram and get the bin counts and edges for punctuation count
human_punc_counts, human_punc_bins = np.histogram(df[df['ind'] == 0]['punc_num'], bins=n_bins)
ai_punc_counts, ai_punc_bins = np.histogram(df[df['ind'] == 1]['punc_num'], bins=n_bins)

# Scale the bin counts to get scaled frequencies
human_scaled_punc_freq = scaler.fit_transform(human_punc_counts.reshape(-1, 1)).flatten()
ai_scaled_punc_freq = scaler.fit_transform(ai_punc_counts.reshape(-1, 1)).flatten()

# Calculate the bin midpoints for plotting
human_punc_bin_midpoints = (human_punc_bins[:-1] + human_punc_bins[1:]) / 2
ai_punc_bin_midpoints = (ai_punc_bins[:-1] + ai_punc_bins[1:]) / 2

# Plotting the scaled frequencies against the bin midpoints
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the scaled frequency histograms
ax.bar(human_punc_bin_midpoints, human_scaled_punc_freq, width=(human_punc_bins[1] - human_punc_bins[0]), alpha=0.5, label='Human Generated', color='blue')
ax.bar(ai_punc_bin_midpoints, ai_scaled_punc_freq, width=(ai_punc_bins[1] - ai_punc_bins[0]), alpha=0.5, label='AI Generated', color='red')

# Add labels and legend
ax.set_xlabel('Punctuation Count')
ax.set_ylabel('Scaled Frequency')
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
```

### 4. Word Count Distribution
   - Created a histogram to compare human-generated vs AI-generated text based on word count.
   ```python
   # Define the number of bins for the histogram
n_bins = 30

# Plot the histogram and get the bin counts and edges
human_counts, human_bins = np.histogram(df[df['ind'] == 0]['word_count'], bins=n_bins)
ai_counts, ai_bins = np.histogram(df[df['ind'] == 1]['word_count'], bins=n_bins)

# Scale the bin counts to get scaled frequencies
human_scaled_freq = scaler.fit_transform(human_counts.reshape(-1, 1)).flatten()
ai_scaled_freq = scaler.fit_transform(ai_counts.reshape(-1, 1)).flatten()

# Calculate the bin midpoints for plotting
human_bin_midpoints = (human_bins[:-1] + human_bins[1:]) / 2
ai_bin_midpoints = (ai_bins[:-1] + ai_bins[1:]) / 2

# Plotting the scaled frequencies against the bin midpoints
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the scaled frequency histograms
ax.bar(human_bin_midpoints, human_scaled_freq, width=(human_bins[1] - human_bins[0]), alpha=0.5, label='Human Generated', color='blue')
ax.bar(ai_bin_midpoints, ai_scaled_freq, width=(ai_bins[1] - ai_bins[0]), alpha=0.5, label='AI Generated', color='red')

# Add labels and legend
ax.set_xlabel('Word Count')
ax.set_ylabel('Scaled Frequency')
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
```

### 5. Spread of Top Features by RFC (Random Forest Classifier)
   - Explored the spread and distribution patterns of the top 8 features (other than word count and punctuation count) identified by the Random Forest Classifier (RFC).
For this part, we need to do some data preparation.

####  **Data Preparation**

The first step is to set predictor variables, X, and target variable, Y.

> Set the target variable (y) = 'ind'

> Set the predictor variable (X) = to the remaining features after dropping 'ind'

> Additionally 'ID' was excluded

####  **Define X and Y variables**

```python
# See, we have 11,144 rows and 772 columns
df.shape

# Looks good! We are looking to have 10X as many rows as columns

# our target variable is 'ind'
# lets set X and y

X = df.drop('ind', axis=1)

y = df['ind']

print(X.shape, y.shape)

X = X.drop('ID', axis=1) # Removing the 'ID' column from X

X.shape

X.head()
```
   #### ** Splitting the dataset into training and testing sets** 
```python
# The test size is set to 0.1, indicating a 90-10 split for training and testing respectively
X_train_plain, X_test_plain, y_train_plain, y_test_plain = train_test_split(X, y,test_size=0.1,random_state=42)

# check to see shape vs our expectations
print(X_train_plain.shape, X_test_plain.shape, y_train_plain.shape, y_test_plain.shape)

# Peak at the first 5 rows
X_train_plain.head()

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Fit a RandomForestClassifier
rf_model = RandomForestClassifier(random_state= 42)
rf_model.fit(X_train_plain, y_train_plain)

# Get feature importances
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train_plain.columns)

# Sort features based on importance
top_features = feature_importances.sort_values(ascending=False).head(10)  # Adjust the number as needed
print("The top 10 features are: \n",top_features)

# make a dataframe with the top 10 most important features
X_top_10= X_train_plain[["feature_44","feature_81","feature_574","feature_283","feature_135","punc_num","feature_174","word_count","feature_586","feature_365"]]

X_top_10.head()

X_top_10.describe

```
After extracting the top 10 features using the Random Forest Classifier (RFC), we discovered that these features encompass word count and punctuation counts. Our current objective is to explore the distribution patterns of these top 10 features. As we've already examined the distribution of both word count and punctuation numbers, we are excluding them from this particular analysis to focus on the remaining eight features within this subset.
```python
# Drop punc_num and word_count from top 10
X_top_8 = X_top_10.drop(["punc_num", "word_count"], axis=1)
```
#### **Boxplot of Top 8 variables on RFC**
```py
# storing the top 8 column names into list
top_8_columns = X_top_8.columns.tolist()

# Plotting boxplots for the top 10 variables
plt.figure(figsize=(10, 6))
X[top_8_columns].boxplot()
plt.title('Boxplots of Top 8 Variables')
plt.xlabel('Variables')
plt.ylabel('Values')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()
```
The provided box plots illustrate the distribution of the top 8 features identified by the Random Forest Classifier (RFC).

In general, the majority of features exhibit values ranging between -2 to 3. However, there is a notable exception observed in 'feature_574', which demonstrates a different distribution pattern. This particular feature showcases a wider range, with its minimum values spanning from -10 to -1, setting it apart from the typical range observed across the other features.
### 6. Correlated Feature Visualization
   - Constructed scatter plots to visualize the relationships and correlations between features, particularly focusing on those features displaying strong correlations. This aided in understanding how certain features interrelate within the dataset.
   - The code conducts a correlation analysis to identify highly correlated features with a correlation coefficient greater than 0.75. Subsequently, it identifies pairs of features displaying significant correlation and proceeds to remove these correlated features from both the training and test datasets. This process aims to reduce the number of features, mitigating multicollinearity concerns and enhancing the dataset's suitability for subsequent modeling or analysis.
```python
# Set the threshold value of 0.75 for correlation
threshold = 0.75

# Calculating the correlation matrix after scaling
correlation_matrix = X_train_plain.corr().abs()

# Finding columns with correlations above the threshold
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Create a set to hold the features to drop
features_to_drop = set()

for column in upper_tri.columns:
    correlated_features = list(upper_tri.index[upper_tri[column] > threshold])
    if correlated_features:
        # Include all highly correlated features above the threshold
        features_to_drop.update(correlated_features[1:])  # Update the set with correlated features except the first one


correlated_pairs = []

for column in upper_tri.columns:
    correlated_features = upper_tri.index[upper_tri[column] > threshold]
    if correlated_features.any():
        # Collect pairs of highly correlated features
        pairs = [(column, corr_feature) for corr_feature in correlated_features]
        correlated_pairs.extend(pairs)

# Convert the list of correlated pairs to a DataFrame
correlated_pairs_df = pd.DataFrame(correlated_pairs, columns=['Feature 1','Feature 2']) # ,'Feature 2']
print(correlated_pairs_df)
```
- The output displayed provides insight into pairs of features that exhibit high correlation. These pairs, identified as highly correlated based on the threshold of 0.75.

- This signifies features within the dataset that showcase strong linear relationships with each other.
```py
num_plots = len(correlated_pairs_df)

# Calculate the number of rows and columns for subplots
num_rows = int(np.ceil(num_plots / 3))  # Increased number of columns per row to 3
num_cols = 3

fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 6 * num_rows))  # Larger figure size

for idx, row in correlated_pairs_df.iterrows():
    feature_1 = row['Feature 1']
    feature_2 = row['Feature 2']
    ax = axes[idx // num_cols, idx % num_cols]

    ax.scatter(X_train_plain[feature_1], X_train_plain[feature_2])
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title(f'Scatterplot of {feature_1} vs {feature_2}')

# Hide empty subplots if any
for i in range(num_plots, num_rows * num_cols):
    axes[i // num_cols, i % num_cols].axis('off')

plt.tight_layout()
plt.show()
```
```py
features_to_drop # Checking the features to drop

# Dropping highly correlated columns from the scaled data
X_train_plain = X_train_plain.drop(columns=features_to_drop, axis=1)
X_test_plain = X_test_plain.drop(columns=features_to_drop, axis=1)

print(X_train_plain.shape)
X_train_plain.head()

print(X_test_plain.shape)
X_test_plain.head()

X_train_plain.shape

X_train_plain.head()

X_test_plain.head()
```
Identification of Highly Correlated Features:

- The correlation matrix (styled_corr) shows pairs of variables that are highly correlated. The specified threshold of 0.75 indicates that variables with a correlation value greater than 0.75 are considered correlated.

- For instance, if a pair such as 'feature_133' and 'feature_167' has a correlation value of 0.751, the two variables are correlated.
Dimension Reduction by Dropping Correlated Features:

- To mitigate multicollinearity and reduce dimensionality, one of the two features from each pair is dropped.

- The code generates a DataFrame, corr_features, containing columns representing the highly correlated features. After removing these correlated features from both the training and test datasets, the final datasets contain 759 features.

- The cleaned dataset has zero null values.

#### Missing Values Analysis
```python

# Missing values by column
df.isnull().sum()

# Missing values in the entire dataframe
df.isnull().sum().sum()
```
## **Feature Engineering**

###  PCA

Principal Componant Analysis - Dimensionality Reduction.

---

Upon implementing Principal Component Analysis (PCA) with the objective of retaining 95% variance, we successfully reduced our dataset from 759 variables to 245 principal components.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # this means keep components that maintain 95% of the variance
X_train_pca = pca.fit_transform(X_train_plain)

# Converting the transformed PCA training data (X_train_pca) to a Pandas DataFrame for further analysis
X_train_pca = pd.DataFrame(X_train_pca)

# Displaying the shape of the PCA-transformed training data (X_train_pca)
print(X_train_pca.shape)

# Displaying the first few rows of the PCA-transformed training data to observe the structure
X_train_pca.head()

# apply that PCA coordinate system
# to the test data
X_test_pca = pca.transform(X_test_plain)

# Converting the transformed PCA test data (X_test_pca) to a Pandas DataFrame for further analysis
X_test_pca = pd.DataFrame(X_test_pca)

# Displaying the shape of the PCA-transformed test data (X_test_pca)
print(X_test_pca.shape)

# Displaying the first few rows of the PCA-transformed test data to observe its structure
X_test_pca.head()

# the y variables are the same as before
y_train_pca = y_train_plain
y_test_pca = y_test_plain
```
## **Modeling**

### Stacked Model
In this section, we present the implementation of a Stacked Model, utilizing an ensemble of various base models to enhance predictive accuracy. The Stacking Classifier combines RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, and SVC models within pipelines, incorporating StandardScaler for effective preprocessing.
``` python
# Importing the important libraries
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier

# Defining the estimators with different models using pipelines with random state = 42
estimators=[
    ( "RandomForestClassifier", make_pipeline(StandardScaler(),
                                          RandomForestClassifier(random_state=42))),
    ( "GradientBoostingClassifier", make_pipeline(StandardScaler(),
                                          GradientBoostingClassifier(random_state=42))),
    ( "AdaBoostClassifier", make_pipeline(StandardScaler(),
                                          AdaBoostClassifier(random_state=42))),
    ( "SVC", make_pipeline(StandardScaler(),
                                          SVC(random_state=42, probability=True)))
]

# Defining StackingClassifier with the above estimators and with final estimator as LogisticRegression
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=42))

clf.fit(X_train_pca, y_train_pca)  # Fitting the model on X_train_pca and y_train_pca
preds = clf.predict(X_test_pca)  # Predicting the values of the X_test_pca and storing the predictions preds

# Evaluate the stacked model's performance (F1-score)
f1_rf = f1_score(y_test_pca, preds)
print(f"F1 Score: {f1_rf:.2f}")

# Get the classification report
report_rf = classification_report(y_test_pca, preds)
print("Classification Report for Stacked Model with Random Forest:\n", report_rf)
```
Ensemble Stacked Model Explanation:

For model construction, an ensemble technique known as the Stacking Classifier was employed to achieve optimal performance. This Stacked Model amalgamates various base models, including RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, and SVC. These models were imported into pipelines along with StandardScaler for effective preprocessing.

The Stacking Classifier's architecture involves combining these base models using their predictions as features in a meta-estimator, Logistic Regression in this case. The stacking approach enhances predictive accuracy by leveraging the collective wisdom of diverse models.

Upon training, the Stacking Classifier learns on the PCA-transformed training data, X_train_pca, and their corresponding labels, y_train_pca. Subsequently, predictions are generated on the PCA-transformed test data, X_test_pca, and these predictions are stored as 'preds'.

### **Evaluation**

#### Confusion Matrix
In this section, we evaluate the Stacked Model's performance using a Confusion Matrix, providing insights into the model's ability to correctly classify instances.

```python
# import confusion matrix library
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test_pca, preds)
print("Confusion Matrix:\n", conf_matrix)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

tn, fp, fn, tp = confusion_matrix(y_test_pca, preds).ravel()
print("TP:", tp)  # print True Positive
print("TN:", tn)  # print True negative
print("FP:", fp)  # print False Positive
print("FN:", fn)  # print False Negative
```
Analysis of the Confusion Matrix from the Stacked Model (y_test_pca vs. preds):

The model accurately predicted 1,004 instances as 0.
It misclassified 6 instances as 1, which were originally labeled as 0.
Additionally, the model incorrectly classified 42 instances as 0 that were originally labeled as 1.
Moreover, it correctly identified 63 instances as 1
```python
# Get the classification report
report_rf = classification_report(y_test_pca, preds)
print("Classification Report for Stacked Model with Random Forest:\n", report_rf)
```
According to the classification report achieved from the stacked model (clf) on PCA variables:

Dataset Distribution: The test dataset consists of 1,010 instances labeled as 0 and 105 instances labeled as 1.

Accuracy: The model achieves an accuracy of 96%, showcasing its overall correct predictions.

Precision: The precision for predicting label 0 is 96%, indicating the model's ability to correctly classify instances as 0. For label 1, the precision is 91%, signifying the model's accuracy in identifying instances as 1. The macro-average precision, which considers both labels equally, stands at 94%. The weighted average precision, accounting for class imbalances, is 96%.

Recall: The recall (sensitivity) for label 0 is 99%, demonstrating the model's capability to capture actual instances labeled as 0. However, for label 1, the recall is 60%, indicating the model's lower performance in correctly identifying instances labeled as 1. The macro-average recall is 80%, and the weighted average recall is 96%.

F1-Score: The F1-score, a balance between precision and recall, for label 0 is 98%, while for label 1, it is 72%. The macro-average F1-score, considering both labels equally, is 85%, and the weighted average F1-score, accounting for class imbalances, stands at 95%.

####  *Permutation Importance*
```python
# Import library for permutation importance
from sklearn.inspection import permutation_importance

# Calculating permutation importance using the classifier (clf),
# the transformed PCA test data (X_test_pca), and corresponding labels (y_test_pca).
# n_repeats define the number of times to permute the feature
result1 = permutation_importance(clf, X_test_pca, y_test_pca, n_repeats=15,
                                  random_state=42)

# Sorting the indices of mean importances obtained from permutation importance
perm_sorted_idx = result1.importances_mean.argsort()

# Get the indices of the top 5 features based on mean importance scores
top_n = 5
top_indices = np.argsort(result1.importances_mean)[-top_n:]

# Get the names of top 5 features
top_feature_names = X_test_pca.columns[top_indices]

print("Top 5 PCA Features:")
print(top_feature_names)
```
After performing permutation importance analysis on PCA-transformed variables using the stacked model that achieved the best F1 score of 0.72,

Top 5 PCA features:
1. 55
2. 5
3. 3
4. 35
5. 1

#### **Box Plots**
```python
# Get the indices of the top 5 features based on mean importance scores
top_n = 5
top_indices = np.argsort(result1.importances_mean)[-top_n:]

# Get the names of top 5 features
top_feature_names = X_test_pca.columns[top_indices]

# Extract permutation importance scores for top features
top_feature_importances = result1.importances[top_indices]

# Create a boxplot for the top 5 features
fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot(top_feature_importances.T, vert=False, labels=top_feature_names)
ax.set_title('Permutation Importance of Top 5 PCA Features')
ax.set_xlabel('Importance Score')
plt.show()
```
The presented boxplot illustrates the permutation importance of the top 5 PCA features. Notably, variable 1 stands out with the highest importance score median, closely hovering around 0.010.
```python
# to retrieve the loadings from PCA variables
loadings = pca.components_

# Creating a DataFrame to store the loadings with column names as original feature names
loadings_df = pd.DataFrame(loadings, columns=X_train_plain.columns)

# Display the loadings
print(loadings_df)

loadings[5]
```
To understand how each original feature contributes to all the principal components, we implemented loadings in PCA variables.
```python
# Update top feature names to reflect the original variables
top_feature_names = X_test_plain.columns[top_indices]

# Extract loadings for the top principal components
top_loadings = loadings[:, top_indices]

# Print the contribution of original features to the top principal components
for i, principal_component_idx in enumerate(top_indices):
    print(f"\nTop Original Features for Principal Component {top_indices[i]}:")

    # Get the loading values for the specified principal component
    pc_loadings = top_loadings[:, i]

    # Identify the top contributing original features (e.g., top 5 features)
    top_contributing_features = X_test_plain.columns[np.argsort(np.abs(pc_loadings))[-5:]]

    # Print the top contributing original features for the current principal component
    print(top_contributing_features)
```
We selected the top 5 original features of each principal component. For example, the top 5 original features of principal component 55 are feature_38, feature_43, feature_177, feature_71, and feature_78.
```python
# Rename PCA columns in X_test_pca from 0,1,2,... to pca_0, pca_1, pca_2,.....respectively
new_column_names = [f'pca_{i}' for i in range(X_test_pca.shape[1])]

# change the column names
X_test_pca.columns = new_column_names
```
 #### **Partial Dependence Plots**
To further understand the relationship between selected PCA variables and the model's predictions, Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) plots were generated.

```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

!pip install pulp
!pip install pycebox
#install ice plot packages
from pycebox.ice import ice, ice_plot
```
#### pca_1
```python
X_train_pca["pca_1"].nunique()

tmpdf = ice(data=X_train_pca,  # ice needs a dataframe
            column='pca_1',    # the column name
            predict=clf.predict,  # the predict statement from the model
            num_grid_points=15)   # setting num_grid points

print(np.shape(tmpdf))

ice_plot(tmpdf, c='dimgray', linewidth=0.3,
         plot_pdp=True,
         pdp_kwargs={'linewidth': 5, 'color':'red'})
plt.title('PDP: pca_1')
plt.ylabel('Predicted ind')
plt.xlabel('pca1')
plt.show()

```
The Partial Dependency plot for variable pca_1 indicates that there are 10,029 unique values. For each unique value in pca_1, as there are 10,029 rows in X_train_pca, the model predicts "ind" 10,029 times. By implementing hyperparameter tuning with num_grid_points=15, we considered only 15 unique values of pca_1. Consequently, we obtain an ICE plot for 15 rows (representing unique rows of pca_1) and 10,029 columns.
#### pca_35
```python
X_train_pca["pca_35"].nunique()

tmpdf35 = ice(data=X_train_pca,  # ice needs a dataframe
              column='pca_35',    # the column name
              predict=clf.predict,  # the predict statement from the model
              num_grid_points=15)

print(np.shape(tmpdf35))

ice_plot(tmpdf35, c='dimgray', linewidth=0.3,
         plot_pdp=True,
         pdp_kwargs={'linewidth': 5, 'color':'red'})
plt.title('PDP: pca_35')
plt.ylabel('Predicted ind')
plt.xlabel('pca_35')
plt.show()
```
In the pca_35 column, there are 10,029 unique values. For each unique value in pca_35, as there are 10,029 rows in X_train_pca, the model predicts "ind" 10,029 times. By implementing hyperparameter tuning with num_grid_points=15, we considered only 15 unique values of pca_35, resulting in an ICE plot for 15 rows (representing unique rows of pca_35) and 10,029 columns.
#### pca_3
```python
X_train_pca["pca_3"].nunique()

tmpdf3 = ice(data=X_train_pca,   # ice needs a dataframe
             column='pca_3',      # the column name
             predict=clf.predict,  # the predict statement from the model
             num_grid_points=15)

print(np.shape(tmpdf3))

ice_plot(tmpdf3, c='dimgray', linewidth=0.3,
         plot_pdp=True,
         pdp_kwargs={'linewidth': 5, 'color':'red'})
plt.title('PDP: pca_3')
plt.ylabel('Predicted ind')
plt.xlabel('pca3')
plt.show()
```
For the pca_3 feature, there are 10,029 unique values. For each unique value in pca_3, as there are 10,029 rows in X_train_pca, the model predicts "ind" 10,029 times. Hyperparameter tuning with num_grid_points=15 intentionally narrows down the focus to just 15 unique values within 'pca_3'. This choice results in the generation of an ICE plot featuring 15 rows (representing unique 'pca_3' values) and 10,029 columns.


## **Conclusion**

- Understanding the Models Predicitions Patterns (3)
### Stacked Model:

- Stacked Models allow for increased model performance. The process follows two steps known as level 0 and level 1.

- At level 0, different models, work together, making predictions about the data.
- Next, the level 1 model is trained using the predictions made by the level 0 models as inputs.
- This technique is effective because each model at level 0 uniquely interprets the data. By combining their predictions, we provide additional insights to the top-level model, enabling it to learn more comprehensively about the patterns and connections in the data.

### PCA-Based Stacked Model:

- Composition: This model employs Principal Components Analysis (PCA), followed by RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, and SupportVectorClassifier as estimators at level 0.
- Level 1 Model: Logistic Regression is used as the meta-model at level 1.
- Performance: The model achieved an F1-score of 0.72
- After identifying the stacked models with the highest F1 scores, we initially implemented permutation importance on 770 variables. Additionally, the stacked model that includes Principal Components Analysis (PCA) also yielded high F1 scores. We then applied permutation importance to the PCA variables, setting 'n_repeats' to 15, to determine their importance.

- To better understand the original features that most contribute to the PCA variables with high importance, we employed loading analysis. This technique reveals how each original feature influences all the principal components. This analysis enabled us to identify the top five original features that significantly contribute to each respective principal component.
