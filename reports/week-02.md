# Week 2 Progress Report

## Project Title

Movie Rating Prediction and Recommendation Using Neural Collaborative Filtering

## What Was Completed This Week

During Week 2, I worked on data preprocessing and baseline model development. The MovieLens 20M dataset was prepared for model training. I loaded the main dataset files, selected the necessary columns, encoded `userId` and `movieId`, and split the dataset into training, validation, and test sets.

I also implemented several simple baseline models for rating prediction. These baseline models are important because they provide comparison results for the deep learning model that will be developed in Week 3.

## Important Commits or Files Changed

Important files created or updated this week:

```text
notebooks/week2_preprocessing_baseline.ipynb
src/preprocessing.py
src/baseline_model.py
src/evaluation.py
reports/week-02.md
results/week2_baseline_results.csv
results/week2_baseline_results.txt
results/week2_example_recommendations.csv
models/week2_baseline_objects.pkl
```

Important commit messages:

```text
add week 2 preprocessing notebook
add user and movie encoding
add train validation test split
add baseline rating prediction model
add RMSE and MAE evaluation
add week 2 progress report
```

## Experiments Run

The following experiments were completed:

```text
loaded ratings.csv and movies.csv
encoded userId and movieId into numerical indexes
created train / validation / test split
tested global average rating baseline
tested movie average rating baseline
tested user average rating baseline
tested combined user-movie average baseline
evaluated models using RMSE and MAE
created a simple recommendation example
```

## Results So Far

The dataset is now ready for model training. The processed data was saved into:

```text
data/processed/train.csv
data/processed/valid.csv
data/processed/test.csv
```

The baseline model results were saved into:

```text
results/week2_baseline_results.csv
results/week2_baseline_results.txt
```

The baseline models show how well simple average-based methods can predict movie ratings. These results will be used as a comparison point for the Neural Collaborative Filtering model in Week 3.

## Problems or Blockers

The main difficulty was the large size of the MovieLens 20M dataset. Processing the full dataset can take a lot of time and memory. To solve this, a sample of the dataset was used for faster Week 2 experiments.

Another challenge is data sparsity. Each user rates only a small number of movies compared to the total number of movies, which makes recommendation more difficult.

## Plan for Next Week

In Week 3, I will start implementing the deep learning model.

Planned tasks:

```text
create PyTorch Dataset and DataLoader
build Neural Collaborative Filtering model
add user embedding layer
add movie embedding layer
train the model
evaluate the model using RMSE and MAE
compare deep learning results with baseline results
save model results
write Week 3 progress report
```

## Conclusion

Week 2 completed the preprocessing and baseline modeling stage. The dataset was encoded, split, and prepared for future training. Baseline rating prediction models were implemented and evaluated. The project is now ready for Week 3 deep learning model development.
