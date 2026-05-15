# Product Recommendation and Rating Prediction Using Amazon Fine Food Reviews

## Project Overview

This project is part of the Applied Deep Learning final project.

The main goal of this project is to build a product recommendation and rating prediction system using the Amazon Fine Food Reviews dataset.

Online stores contain thousands of products, so users may have difficulty finding products that match their interests. Recommendation systems help solve this problem by predicting which products a user may like.

In this project, the model will use customer review data, product information, user information, rating scores, and review text to predict product ratings or classify whether a user likes a product or not.

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

The dataset contains Amazon food product reviews. It includes product information, user information, rating scores, review summaries, and full review text.

## Main Columns

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

## Target Column

```text
Score
```

For classification, the target will be converted into:

```text
Like / Not Like
```

## Planned Method

### Baseline Model

The baseline model will be Logistic Regression.

The review text will be converted into numerical features using TF-IDF. Then Logistic Regression will classify reviews into two classes:

- Like
- Not Like

The baseline model will be used for comparison with the deep learning model.

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

LSTM is suitable for this task because review text is sequential data. The model can learn important patterns from words and phrases in customer reviews.

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

The validation set will be used to tune model parameters and compare models.

The test set will be used only for final evaluation.

## Repository Structure

```text
amazon-fine-food-recommendation/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── .gitkeep
├── src/
│   └── explore_data.py
├── reports/
│   └── week-01.md
├── results/
│   └── .gitkeep
└── final-report.md
```

## Week 1 Work

During Week 1, the repository was created and the dataset was selected. The main focus was project setup and dataset understanding.

Completed tasks:

- created GitHub repository;
- created project folder structure;
- added README.md;
- added dataset instructions;
- added requirements.txt;
- added .gitignore;
- prepared Week 1 progress report;
- prepared basic dataset exploration script.

## How to Run Week 1 Exploration

### 1. Clone the repository

```bash
git clone https://github.com/your-username/amazon-fine-food-recommendation.git
cd amazon-fine-food-recommendation
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

### 4. Install libraries

```bash
pip install -r requirements.txt
```

### 5. Download dataset

Download the dataset from Kaggle:

```text
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
```

Place the file here:

```text
data/Reviews.csv
```

Large dataset files should not be committed to GitHub.

### 6. Run dataset exploration

```bash
python src/explore_data.py
```

## Expected Final Output

The final project will include:

- dataset loading instructions;
- data preprocessing code;
- baseline model;
- deep learning model;
- evaluation code;
- weekly progress reports;
- final report;
- result tables and visualizations.

## References

1. Kaggle: Amazon Fine Food Reviews  
   https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

2. SNAP Stanford: Amazon Fine Food Reviews  
   https://snap.stanford.edu/data/web-FineFoods.html
