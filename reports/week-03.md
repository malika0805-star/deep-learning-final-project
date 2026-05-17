# Week 3 Progress Report

## Project Title

Movie Rating Prediction and Recommendation Using Neural Collaborative Filtering

---

## 1. What Was Completed This Week

During Week 3, the main focus was implementing the deep learning part of the recommendation system. In the previous week, the dataset was preprocessed, user and movie IDs were encoded, and baseline models were created. This week, I used the processed data to build and train a Neural Collaborative Filtering model.

The main task was to create a model that can learn hidden relationships between users and movies. Unlike the baseline models, which only use simple averages, the deep learning model learns user and movie embeddings. These embeddings represent users and movies as numerical vectors and help the model predict ratings more flexibly.

The following main tasks were completed:

```text
loaded processed train, validation, and test datasets
created PyTorch Dataset class
created DataLoader objects for batch training
built Neural Collaborative Filtering model
added user embedding layer
added movie embedding layer
added fully connected neural network layers
trained the model using MSE Loss
evaluated the model using RMSE and MAE
saved training history
saved deep learning results
saved model comparison file
saved trained model file
```

---

## 2. Important Commits or Files Changed

The following files were created or updated during Week 3:

```text
notebooks/week3_deep_learning_model.ipynb
src/deep_learning_model.py
reports/week-03.md
results/week3_deep_learning_results.csv
results/week3_deep_learning_results.txt
results/week3_training_history.csv
results/week3_model_comparison.csv
models/week3_neural_collaborative_filtering.pt
```

The notebook contains the step-by-step experiment. The `src/deep_learning_model.py` file contains the cleaner script version of the model training process.

Important commit messages for Week 3:

```text
add week 3 deep learning notebook
add neural collaborative filtering model
add PyTorch dataset and dataloader
add training loop and validation evaluation
add deep learning results
add model comparison file
add week 3 progress report
```

---

## 3. Main Tasks Completed

### 3.1 Loading Processed Data

The model used the processed files from Week 2:

```text
data/processed/train.csv
data/processed/valid.csv
data/processed/test.csv
```

These files already included encoded columns:

```text
user_index
movie_index
rating
```

The columns `user_index` and `movie_index` were used as model inputs, and the `rating` column was used as the prediction target.

---

### 3.2 Creating PyTorch Dataset and DataLoader

A custom PyTorch Dataset class was created to store user indexes, movie indexes, and ratings. This made the data suitable for training with PyTorch.

The DataLoader was used to divide the data into batches. This is important because the dataset is large, and training on the full dataset at once is inefficient.

The batch size used was:

```text
4096
```

The training DataLoader used shuffling, while validation and test DataLoaders did not use shuffling.

---

### 3.3 Building the Neural Collaborative Filtering Model

The main deep learning model was Neural Collaborative Filtering.

The model structure was:

```text
user_index input
      ↓
User Embedding
      ↓
Concatenation
      ↓
Fully Connected Layers
      ↓
Predicted Rating
      ↑
Movie Embedding
      ↑
movie_index input
```

The model included:

```text
User Embedding Layer
Movie Embedding Layer
Concatenation Layer
Fully Connected Layers
Dropout Layers
Output Layer
```

The user embedding learns hidden user preferences. The movie embedding learns hidden movie characteristics. After that, the embeddings are concatenated and passed through fully connected layers to predict the rating.

---

### 3.4 Training the Model

The model was trained using:

```text
Mean Squared Error Loss
Adam Optimizer
```

MSE Loss was selected because the task is rating prediction, not classification. The model predicts a numerical rating value, so the training objective is to reduce the difference between the real rating and the predicted rating.

The training process included:

```text
forward pass
loss calculation
backpropagation
optimizer update
validation after each epoch
```

The model was trained for several epochs, and after each epoch, validation RMSE and MAE were calculated.

---

### 3.5 Evaluation

The model was evaluated using:

```text
RMSE
MAE
```

RMSE gives stronger penalty to large prediction errors. MAE shows the average absolute difference between real and predicted ratings.

Predicted ratings were clipped to the valid MovieLens rating range:

```text
0.5 to 5.0
```

This prevents impossible predictions such as ratings below 0.5 or above 5.0.

---

## 4. Experiments Run

The following experiments were run during Week 3:

```text
trained Neural Collaborative Filtering model with user and movie embeddings
tested validation RMSE after each epoch
tested validation MAE after each epoch
evaluated final model on the test set
compared deep learning model with Week 2 baseline models
saved training history for analysis
```

The model was compared with the baseline results from Week 2. This comparison is important because it shows whether the deep learning model improves rating prediction compared to simple average-based methods.

---

## 5. Results So Far

By the end of Week 3, the project has a complete deep learning model for rating prediction.

The deep learning results were saved in:

```text
results/week3_deep_learning_results.csv
results/week3_deep_learning_results.txt
```

The training history was saved in:

```text
results/week3_training_history.csv
```

The comparison between baseline and deep learning models was saved in:

```text
results/week3_model_comparison.csv
```

The trained model was saved in:

```text
models/week3_neural_collaborative_filtering.pt
```

The current project pipeline now includes:

```text
data preprocessing
baseline model
deep learning model
evaluation
model comparison
```

This means the project has moved from preparation and baseline testing to the main deep learning implementation stage.

---

## 6. Problems or Blockers

The main challenge this week was training time. Deep learning models take more time to train than simple baseline models, especially when the dataset is large.

Another challenge was memory usage. MovieLens 20M is a large dataset, so training with the full dataset may not be efficient on a normal laptop. For this reason, using a sample of the data during development is reasonable.

There is also a risk of overfitting. If the model becomes too complex, it may perform well on training data but worse on validation and test data. To reduce this risk, dropout layers were added, and validation RMSE/MAE were monitored during training.

---

## 7. Plan for Next Week

In Week 4, the project will focus on final evaluation, interpretation, and reporting.

Planned tasks for Week 4:

```text
compare baseline and deep learning model results
analyze prediction errors
identify cases where the model performs well and poorly
create final results table
prepare model comparison explanation
update final report
prepare presentation or demo
write Week 4 progress report
```

The final week will also include explaining the limitations of the recommendation system, such as sparse user-movie interactions and the cold-start problem.

---

## 8. Conclusion

Week 3 was focused on implementing the main deep learning model. The Neural Collaborative Filtering model was built using user embeddings, movie embeddings, and fully connected layers.

The model was trained and evaluated using RMSE and MAE. The training history, final results, and model comparison files were saved. The project now has both baseline and deep learning results, which makes it ready for final evaluation and report preparation in Week 4.
