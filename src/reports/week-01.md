# Week 1 Progress Report

## Project Title

Product Recommendation and Rating Prediction Using Amazon Fine Food Reviews

## Week

Week 1

## Main Goal of the Week

The main goal of Week 1 was to start the final project, choose a suitable dataset, create the GitHub repository, and prepare the basic project structure. This week focused on project setup and dataset understanding before starting the main implementation.

## Completed Work

This week, I selected the project topic and dataset. The selected topic is product recommendation and rating prediction using the Amazon Fine Food Reviews dataset.

I created the GitHub repository and organized the project folders. I added the main project files such as README.md, requirements.txt, .gitignore, data instructions, source code folder, reports folder, results folder, and final report template.

I also prepared a basic dataset exploration script. This script will help to check the dataset shape, column names, missing values, rating distribution, and binary Like / Not Like class distribution.

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

Dataset name: Amazon Fine Food Reviews

Dataset source:

- Kaggle: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
- SNAP Stanford: https://snap.stanford.edu/data/web-FineFoods.html

Main file:

```text
Reviews.csv
```

The dataset contains Amazon food product reviews. It includes user information, product information, rating scores, review summaries, and full review text.

## Important Columns

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

## Files Created This Week

The following files and folders were created during Week 1:

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

## Important Files Changed

| File | Purpose |
|---|---|
| README.md | Main project description and instructions |
| .gitignore | Prevents large dataset files and unnecessary files from being committed |
| requirements.txt | Lists required Python libraries |
| data/README.md | Explains how to download and place the dataset |
| src/explore_data.py | Basic dataset exploration script |
| reports/week-01.md | Week 1 progress report |
| final-report.md | Initial final report template |

## Experiments Run

This week, no training experiments were completed yet. The main focus was on project setup and dataset preparation.

The planned dataset exploration script checks:

- dataset shape;
- first rows of the dataset;
- column names;
- missing values;
- rating distribution;
- binary Like / Not Like distribution.

## Basic Exploration Script

The file created for basic dataset exploration is:

```text
src/explore_data.py
```

This script will load the dataset from:

```text
data/Reviews.csv
```

It will generate basic information about the dataset and save simple results into the `results/` folder.

Expected output files:

```text
results/rating_distribution.png
results/like_distribution.png
results/week1_dataset_summary.txt
```

## Results So Far

The project repository is ready for further development.

The dataset source is selected and documented. The main project structure is created. The Week 1 progress report is completed. The project is ready for Week 2, where data preprocessing and baseline model training will begin.

## Problems or Blockers

The main issue is that the dataset file is large, so it should not be uploaded directly to GitHub. To solve this, the dataset will be downloaded manually from Kaggle and placed inside the `data/` folder.

The file `Reviews.csv` is ignored by `.gitignore`.

## Planned Commits for Week 1

Meaningful commits for Week 1:

```text
add project structure
add dataset instructions
add requirements and gitignore
update README with project overview
add dataset exploration script
add week 1 progress report
add final report template
```

## Plan for Next Week

Next week, I will start data preprocessing.

Planned tasks for Week 2:

- load the dataset;
- select important columns;
- remove missing values;
- clean review text;
- create binary target label;
- split data into training, validation, and test sets;
- save processed data;
- train Logistic Regression baseline model;
- evaluate baseline model using accuracy, precision, recall, F1-score, and confusion matrix;
- write Week 2 progress report.

## Conclusion

Week 1 was focused on project preparation. The topic and dataset were selected, the GitHub repository structure was created, and the first documentation files were prepared. The project is now ready for the next step: data preprocessing and baseline model development.
