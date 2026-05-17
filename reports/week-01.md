# Week 1 Progress Report

## Project Title

Movie Rating Prediction and Recommendation Using Neural Collaborative Filtering

---

## 1. Work Completed This Week

During the first week, the main focus was on preparing the project foundation. I selected the project direction, chose a suitable dataset, created the GitHub repository, and organized the initial folder structure.

The selected task is **recommendation / ranking**. The project will use the **MovieLens 20M** dataset to predict movie ratings for users and then rank movies according to predicted ratings.

The repository was prepared with the required folders for data, notebooks, source code, weekly reports, results, and trained models. I also added the main project documentation files, including `README.md`, `.gitignore`, `requirements.txt`, and `final-report.md`.

---

## 2. Dataset and Task Description

The dataset selected for this project is:

```text
MovieLens 20M
```

Dataset source:

```text
https://grouplens.org/datasets/movielens/20m/
```

The main files used in the project are:

```text
ratings.csv
movies.csv
```

The recommendation task is based on user-movie interactions.

Model input:

```text
userId
movieId
```

Model output:

```text
predicted rating
```

The predicted rating will later be used to rank movies and recommend the most suitable movies to each user.

---

## 3. Important Files Created

The following files and folders were prepared during Week 1:

```text
README.md
.gitignore
requirements.txt
data/.gitkeep
notebooks/.gitkeep
src/explore_data.py
src/preprocessing.py
src/baseline_model.py
src/deep_learning_model.py
src/evaluation.py
reports/week-01.md
results/.gitkeep
models/.gitkeep
final-report.md
```

The file `src/explore_data.py` was added for basic dataset exploration.  
The files `src/preprocessing.py`, `src/baseline_model.py`, and `src/deep_learning_model.py` were added as placeholders for the next stages of the project.

---

## 4. Important Commits

Meaningful commits for Week 1 include:

```text
add empty project folders
add project README
add gitignore file
add requirements file
add dataset exploration script
add source code placeholders
add evaluation metrics functions
add week 1 progress report
add final report template
```

These commits show gradual project progress instead of uploading everything at the end.

---

## 5. Experiments Run

No model training was performed during Week 1.

The main technical work was preparing the dataset exploration script. The script is designed to check:

```text
dataset shape
column names
missing values
rating distribution
number of unique users
number of unique movies
```

This exploration step is important because it helps understand the structure of the dataset before preprocessing and model training.

---

## 6. Results So Far

By the end of Week 1, the project repository is ready for further development.

The project topic is clearly defined, the dataset has been selected, and the main project files have been created. The repository now has a clean structure that can support the next steps of the project.

The current project direction is:

```text
Task type: Recommendation / Ranking
Dataset: MovieLens 20M
Prediction target: rating
Evaluation metrics: RMSE and MAE
```

---

## 7. Problems or Blockers

The main issue is the size of the MovieLens 20M dataset. Since the dataset is large, raw files should not be uploaded directly to GitHub.

The following files will be stored locally only:

```text
ratings.csv
movies.csv
ml-20m.zip
```

To prevent large files from being committed, `.gitignore` was added to the repository.

---

## 8. Plan for Next Week

In Week 2, the project will move from setup to implementation.

Planned tasks for Week 2:

```text
load ratings.csv and movies.csv
check missing values
encode userId and movieId
split the dataset into train, validation, and test sets
build a baseline rating prediction model
evaluate the baseline model using RMSE and MAE
save baseline results
prepare Week 2 progress report
```

---

## 9. Conclusion

Week 1 was mainly focused on project preparation. The repository structure was created, the dataset was selected, and the first source code files were prepared.

The project is now ready for Week 2, where preprocessing and baseline model development will begin.
