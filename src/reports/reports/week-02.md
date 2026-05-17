# Week 2 Progress Report

## Project Title

Product Recommendation and Rating Prediction Using Amazon Fine Food Reviews

## Week

Week 2

## Main Goal of the Week

The main goal of Week 2 was to prepare the dataset for machine learning and create the first baseline model. This week focused on loading the dataset, cleaning the review text, creating the target label, splitting the data into training, validation, and test sets, and preparing a baseline Logistic Regression model.

This week is important because before using a deep learning model, the dataset must be cleaned and converted into a format that models can understand. Also, the baseline model is needed to compare future deep learning results.

## Completed Work

During Week 2, I worked on data preprocessing and baseline model preparation.

The following tasks were completed:

- loaded the Amazon Fine Food Reviews dataset;
- selected important columns from the dataset;
- checked missing values;
- removed rows with missing important values;
- cleaned review text;
- created a binary target label;
- split the dataset into training, validation, and test sets;
- created evaluation functions;
- created Logistic Regression baseline model;
- prepared result saving system;
- updated README.md with Week 2 instructions.

## Project Task

The project is based on a recommendation / ranking and rating prediction task.

However, during Week 2, the task was simplified into a binary classification problem. The model predicts whether a user likes a product or not based on the review text.

The rating score was converted into two classes:

```text
Score >= 4  -> Like = 1
Score < 4   -> Not Like = 0
```

This allows the project to use classification metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

## Dataset Used

Dataset name:

```text
Amazon Fine Food Reviews
```

Dataset source:

```text
Kaggle: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
SNAP Stanford: https://snap.stanford.edu/data/web-FineFoods.html
```

Main dataset file:

```text
Reviews.csv
```

Dataset path in the project:

```text
data/Reviews.csv
```

The dataset contains Amazon food product reviews. Each row represents one customer review. The dataset includes user information, product information, rating score, review summary, and full review text.

## Columns Used

The following columns were selected for Week 2:

| Column | Description |
|---|---|
| Id | Unique review ID |
| ProductId | Product identifier |
| UserId | User identifier |
| Score | Rating from 1 to 5 |
| Summary | Short review summary |
| Text | Full review text |

The most important columns for Week 2 are:

```text
Text
Score
```

The `Text` column is used as input data.

The `Score` column is used to create the target label.

## Target Label

A new target column was created:

```text
Like
```

The rule for creating this label is:

```text
Score >= 4  -> Like = 1
Score < 4   -> Like = 0
```

This means:

| Score | Class |
|---|---|
| 1 | Not Like |
| 2 | Not Like |
| 3 | Not Like |
| 4 | Like |
| 5 | Like |

This target label makes the task suitable for binary classification.

## Data Preprocessing Steps

The preprocessing script is located in:

```text
src/preprocessing.py
```

The script performs the following steps:

```text
1. Load the dataset from data/Reviews.csv
2. Select important columns
3. Remove rows with missing values
4. Clean the review text
5. Create the binary target label
6. Split the data into train, validation, and test sets
7. Save processed datasets
```

## Text Cleaning

The review text was cleaned before model training.

The cleaning process included:

- converting all text to lowercase;
- removing HTML tags;
- removing numbers and special symbols;
- removing extra spaces;
- converting all text into a cleaner format.

Example of text cleaning:

```text
Original text:
"This Product is GREAT!!! <br /> I loved it."

Cleaned text:
"this product is great i loved it"
```

Text cleaning is important because raw reviews often contain unnecessary symbols, HTML tags, capital letters, punctuation, and other noise.

## Dataset Sampling

The full dataset is large, so for faster experiments during Week 2, a sample of the dataset can be used.

Sample size used:

```text
100,000 reviews
```

This helps to reduce training time while still keeping the dataset large enough for meaningful training and evaluation.

Later, the sample size can be increased, or the full dataset can be used for final experiments.

## Train / Validation / Test Split

The dataset was split into three parts:

| Dataset Part | Percentage | Purpose |
|---|---:|---|
| Training set | 70% | Used to train the model |
| Validation set | 15% | Used to check model performance during development |
| Test set | 15% | Used for final model evaluation |

The split was stratified by the binary label `Like`.

This means the proportion of Like and Not Like classes stays similar in training, validation, and test sets.

Expected output files after preprocessing:

```text
data/processed/train.csv
data/processed/valid.csv
data/processed/test.csv
```

## Baseline Model

The baseline model for Week 2 is:

```text
Logistic Regression
```

Logistic Regression was selected because it is a simple and understandable machine learning model. It is useful as a baseline before training a deep learning model.

The baseline model uses review text as input.

Model pipeline:

```text
Clean review text
        ↓
TF-IDF Vectorizer
        ↓
Logistic Regression
        ↓
Like / Not Like prediction
```

## TF-IDF Vectorization

Machine learning models cannot directly understand raw text. Therefore, the review text was converted into numerical features using TF-IDF.

TF-IDF means:

```text
Term Frequency - Inverse Document Frequency
```

It gives higher importance to words that are meaningful in a review and lower importance to words that appear too often.

Example:

```text
Review text:
"this coffee is very good and delicious"

TF-IDF converts this sentence into numerical features.
```

TF-IDF settings used in the baseline model:

```text
max_features = 20000
min_df = 2
max_df = 0.90
ngram_range = (1, 2)
```

Explanation:

| Parameter | Meaning |
|---|---|
| max_features | Maximum number of words/features |
| min_df | Ignore words that appear too rarely |
| max_df | Ignore words that appear too frequently |
| ngram_range | Use single words and two-word combinations |

## Evaluation Metrics

The baseline model was evaluated using the required classification metrics:

```text
Accuracy
Precision
Recall
F1-score
Confusion matrix
```

### Accuracy

Accuracy shows the percentage of correct predictions.

```text
Accuracy = correct predictions / all predictions
```

### Precision

Precision shows how many predicted positive examples were actually positive.

```text
Precision = true positives / predicted positives
```

### Recall

Recall shows how many actual positive examples were found by the model.

```text
Recall = true positives / actual positives
```

### F1-score

F1-score is a balance between precision and recall.

```text
F1-score = harmonic mean of precision and recall
```

### Confusion Matrix

The confusion matrix shows correct and incorrect predictions for both classes.

```text
               Predicted Not Like    Predicted Like
Actual Not Like        TN                 FP
Actual Like            FN                 TP
```

Where:

| Term | Meaning |
|---|---|
| TN | True Negative |
| FP | False Positive |
| FN | False Negative |
| TP | True Positive |

## Regression Metrics

For Week 2, the main task is binary classification.

However, the project can also include rating prediction later. In that case, the following metrics will be used:

```text
RMSE
MAE
```

RMSE and MAE are useful when the model predicts the exact rating value from 1 to 5.

## Files Created or Updated This Week

| File | Purpose |
|---|---|
| src/preprocessing.py | Loads, cleans, labels, and splits the dataset |
| src/evaluation.py | Contains evaluation metric functions |
| src/baseline_model.py | Trains Logistic Regression baseline model |
| requirements.txt | Adds required Python libraries |
| .gitignore | Ignores raw dataset, processed data, and model files |
| README.md | Adds Week 2 instructions |
| reports/week-02.md | Week 2 progress report |

## Code Files Description

### src/preprocessing.py

This file is responsible for data preparation.

Main tasks:

```text
Load dataset
Select useful columns
Remove missing values
Clean review text
Create Like label
Split data
Save processed files
```

### src/evaluation.py

This file contains reusable evaluation functions.

Main tasks:

```text
Calculate accuracy
Calculate precision
Calculate recall
Calculate F1-score
Create confusion matrix
Calculate RMSE and MAE if needed
Save metrics
Save confusion matrix
```

### src/baseline_model.py

This file trains the first baseline model.

Main tasks:

```text
Load processed data
Convert text into TF-IDF features
Train Logistic Regression
Evaluate validation set
Evaluate test set
Save results
Save trained model
```

## How to Run Week 2 Code

Before running the code, the dataset file must be placed here:

```text
data/Reviews.csv
```

### Step 1: Run preprocessing

```bash
python src/preprocessing.py
```

Expected output:

```text
data/processed/train.csv
data/processed/valid.csv
data/processed/test.csv
```

### Step 2: Train baseline model

```bash
python src/baseline_model.py
```

Expected output:

```text
results/baseline_test_metrics.txt
results/baseline_confusion_matrix.png
models/baseline_logistic_regression.pkl
models/tfidf_vectorizer.pkl
```

## Expected Results

After running the baseline model, the project should produce:

- baseline classification metrics;
- confusion matrix image;
- saved Logistic Regression model;
- saved TF-IDF vectorizer;
- processed train/validation/test datasets.

Expected results folder:

```text
results/
├── baseline_test_metrics.txt
└── baseline_confusion_matrix.png
```

Expected models folder:

```text
models/
├── baseline_logistic_regression.pkl
└── tfidf_vectorizer.pkl
```

## Results So Far

By the end of Week 2, the project has a complete data preprocessing pipeline and a baseline machine learning model.

The dataset can now be loaded, cleaned, split, and used for training. The baseline model gives the first performance results and creates a comparison point for the deep learning model in Week 3.

The exact numerical results will depend on the sample size and computer environment.

## Problems or Blockers

One challenge is dataset size. The Amazon Fine Food Reviews dataset is large, so preprocessing and training may take time. To solve this, a sample of 100,000 reviews is used during initial experiments.

Another possible issue is class imbalance. Many Amazon reviews have high scores, so the Like class may be larger than the Not Like class. To reduce this problem, class balancing is used in Logistic Regression.

Text data can also be noisy. Reviews may contain spelling mistakes, symbols, HTML tags, very short comments, or repeated words. Text cleaning helps reduce this issue.

## Important Commits for Week 2

Meaningful commit messages for Week 2:

```text
update requirements for week 2
add preprocessing script
add evaluation functions
add baseline logistic regression model
update gitignore for processed data and models
update README with week 2 instructions
add week 2 progress report
```

## Plan for Next Week

In Week 3, I will start working on the deep learning model.

Planned tasks for Week 3:

- prepare text data for deep learning;
- create vocabulary or tokenizer;
- convert review text into sequences;
- build an LSTM model;
- train the LSTM model;
- evaluate the model on validation and test data;
- compare LSTM results with Logistic Regression baseline;
- save training loss plots;
- save final metrics;
- write Week 3 progress report.

## Conclusion

Week 2 focused on data preprocessing and baseline model development. The dataset was prepared for machine learning, a binary target label was created, and the first baseline model was implemented.

The Logistic Regression baseline provides a simple comparison point for future deep learning models. The project is now ready for Week 3, where an LSTM-based deep learning model will be developed and compared with the baseline.
