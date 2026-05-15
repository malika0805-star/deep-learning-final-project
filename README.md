# Product Recommendation and Rating Prediction Using Amazon Fine Food Reviews

## Project Overview

This project is part of the Applied Deep Learning final project. The main goal is to build a recommendation and rating prediction system using the Amazon Fine Food Reviews dataset.

Online stores contain thousands of products, so users may have difficulty finding products that match their interests. Recommendation systems help solve this problem by predicting which products a user may like.

In this project, the model will use customer review data, product information, user information, and review text to predict product ratings or classify whether a user likes a product or not.

## Project Task

The project is based on a recommendation / ranking task.

The model will predict:

- the rating score given by a user to a product;
- or whether the user likes the product.

For binary classification, the rating score will be converted into two classes:

```text
Score >= 4  -> Like = 1
Score < 4   -> Not Like = 0
```

## Dataset

Dataset name: **Amazon Fine Food Reviews**

Dataset source:

- Kaggle: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
- SNAP Stanford: https://snap.stanford.edu/data/web-FineFoods.html

Main file used:

```text
Reviews.csv
```

The dataset contains Amazon food product reviews. It includes product information, user information, rating scores, and review text.

Main columns used in this project:

| Column | Description |
|---|---|
| Id | Review ID |
| ProductId | Product identifier |
| UserId | User identifier |
| ProfileName | User profile name |
| HelpfulnessNumerator | Number of users who found the review helpful |
| HelpfulnessDenominator | Number of users who voted on helpfulness |
| Score | Rating from 1 to 5 |
| Time | Review timestamp |
| Summary | Short review summary |
| Text | Full review text |

Target column:

```text
Score
```

For classification, the target will be converted into:

```text
Like / Not Like
```

## Repository Structure

```text
project-repo/
├── README.md
├── data/
│   └── README.md
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── preprocessing.py
│   ├── baseline_model.py
│   ├── deep_learning_model.py
│   └── evaluation.py
├── reports/
│   ├── week-01.md
│   ├── week-02.md
│   ├── week-03.md
│   └── week-04.md
├── results/
│   ├── figures/
│   └── metrics.csv
├── requirements.txt
└── final-report.md
```

## Planned Method

### Baseline Model

The baseline model will be Logistic Regression.

The review text will be converted into numerical features using TF-IDF. Then Logistic Regression will classify reviews into two classes:

- Like
- Not Like

The baseline model is used for comparison with the deep learning model.

### Deep Learning Model

The main deep learning model will be an LSTM-based neural network.

The model will include:

```text
Text input
Embedding layer
LSTM layer
Fully connected layer
Output layer
```

LSTM is suitable for this task because review text is sequential data. The model can learn patterns from words and phrases in customer reviews.

Possible positive words:

```text
good, great, delicious, excellent, love
```

Possible negative words:

```text
bad, disappointed, terrible, waste, poor
```

## Loss Functions

For binary classification:

```text
Binary Cross-Entropy Loss
```

For rating prediction:

```text
Mean Squared Error Loss
```

## Evaluation Metrics

The project will use the following evaluation metrics.

For classification:

```text
Accuracy
Precision
Recall
F1-score
Confusion matrix
```

For rating prediction:

```text
RMSE
MAE
```

BLEU will not be used because it is mainly used for translation and text generation tasks.

## Train / Validation / Test Split

The dataset will be split into three parts:

| Dataset Part | Percentage |
|---|---|
| Training set | 70% |
| Validation set | 15% |
| Test set | 15% |

The training set will be used to train the model.

The validation set will be used to tune parameters and compare models.

The test set will be used only for final evaluation.

## Expected Challenges

Possible challenges in this project:

- The dataset may be imbalanced because many reviews have high ratings.
- Review text may contain spelling mistakes, symbols, and noisy words.
- The dataset is large, so preprocessing and training may take time.
- LSTM training may require more memory and time.
- The model may overfit if it becomes too complex.

## Weekly Plan

| Week | Planned Work | Expected Output |
|---|---|---|
| Week 1 | Dataset selection, GitHub repository setup, exploratory data analysis | Proposal, README, dataset summary |
| Week 2 | Data preprocessing, text cleaning, train/validation/test split, baseline model | Cleaned data, baseline results, Week 2 report |
| Week 3 | Deep learning model training using LSTM, experiments, loss curves | Model results, plots, error analysis |
| Week 4 | Final improvements, final evaluation, final report and presentation | Final code, final report, presentation/demo |

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate virtual environment

For macOS / Linux:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 4. Install required libraries

```bash
pip install -r requirements.txt
```

### 5. Download the dataset

Download the dataset from Kaggle:

```text
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
```

Place the file in the `data/` folder:

```text
data/Reviews.csv
```

Large dataset files should not be committed to GitHub.

### 6. Run preprocessing

```bash
python src/preprocessing.py
```

### 7. Train baseline model

```bash
python src/baseline_model.py
```

### 8. Train deep learning model

```bash
python src/deep_learning_model.py
```

### 9. Evaluate model

```bash
python src/evaluation.py
```

## Expected Results

The final result will include:

- cleaned dataset;
- baseline model results;
- LSTM model results;
- comparison of models;
- evaluation metrics;
- confusion matrix;
- RMSE and MAE values;
- final report;
- project presentation.

## Final Report

The final report will include:

- project title;
- problem statement;
- dataset description;
- data preprocessing;
- model architecture;
- training setup;
- evaluation metrics;
- results table;
- error analysis;
- limitations;
- conclusion;
- references.

## References

1. Amazon Fine Food Reviews Dataset, Kaggle: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
2. SNAP Stanford Amazon Fine Food Reviews: https://snap.stanford.edu/data/web-FineFoods.html
