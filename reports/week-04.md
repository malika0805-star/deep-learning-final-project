# Week 4 Progress Report

## Project Title

Movie Rating Prediction and Recommendation Using Neural Collaborative Filtering

---

## What Was Completed This Week

During Week 4, the project was finalized. The main focus was final evaluation, model comparison, error analysis, and preparation of the final project outputs.

I compared the baseline models from Week 2 with the Neural Collaborative Filtering model from Week 3. The comparison was based on RMSE and MAE because the project is a rating prediction and recommendation task.

I also performed error analysis by comparing real ratings with predicted ratings. This helped identify examples where the model predicted accurately and examples where the prediction error was high. Finally, I created a simple Top-10 recommendation demo for one user.

---

## Important Commits or Files Changed

Important files created or updated this week:

```text
notebooks/week4_final_evaluation.ipynb
reports/week-04.md
results/final_model_comparison.csv
results/week4_final_rmse_comparison.png
results/week4_final_mae_comparison.png
results/week4_error_analysis.csv
results/week4_best_predictions.csv
results/week4_worst_predictions.csv
results/week4_top10_recommendations.csv
results/week4_final_summary.txt
```

Important commit messages:

```text
add final evaluation notebook
add final model comparison
add error analysis results
add top 10 recommendation demo
add week 4 progress report
update final project results
```

---

## Experiments Run

The following experiments were completed during Week 4:

```text
loaded Week 2 baseline results
loaded Week 3 deep learning results
created final model comparison table
identified the best model based on RMSE and MAE
loaded the trained Neural Collaborative Filtering model
generated predictions on the test set
calculated prediction errors
created best and worst prediction examples
created Top-10 recommendation example
saved final summary files
```

The final evaluation used the test set and the main metrics:

```text
RMSE
MAE
```

---

## Results So Far

The final comparison table was saved as:

```text
results/final_model_comparison.csv
```

The error analysis was saved as:

```text
results/week4_error_analysis.csv
results/week4_best_predictions.csv
results/week4_worst_predictions.csv
```

The recommendation demo was saved as:

```text
results/week4_top10_recommendations.csv
```

The project now includes a complete workflow:

```text
dataset exploration
data preprocessing
baseline model
deep learning model
final model comparison
error analysis
recommendation demo
```

---

## Problems or Blockers

The main challenge was working with a large dataset and a trained deep learning model. Loading the model and generating predictions can take time, especially if the computer does not have a GPU.

Another challenge is that the Neural Collaborative Filtering model may not always outperform the strongest baseline model. This can happen if the model is trained for a limited number of epochs, uses a sample of the dataset, or needs more hyperparameter tuning.

This result is still useful because it gives a fair comparison between simple baseline models and a deep learning approach.

---

## Final Output

The final project includes:

```text
GitHub repository
README.md
dataset instructions
source code
notebooks
weekly progress reports
baseline model results
deep learning model results
final model comparison
error analysis
Top-10 recommendation example
final report
```

The final files prepared in Week 4 will be used in the final report and presentation.

---

## Plan After Week 4

After Week 4, the remaining work is to finalize the written report and presentation/demo.

The final report should include:

```text
project title
problem statement
dataset description
data preprocessing
baseline model
deep learning model
training setup
evaluation metrics
results table
error analysis
limitations
conclusion
references
```

The presentation should briefly explain the problem, dataset, model approach, results, and limitations.

---

## Conclusion

Week 4 completed the final evaluation stage of the project. The baseline and deep learning models were compared, prediction errors were analyzed, and a Top-10 recommendation example was created.

The project is now ready for final report writing and presentation.
