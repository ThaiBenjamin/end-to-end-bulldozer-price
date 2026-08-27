# End-to-End Bulldozer Price Regression

Predicting the auction sale price of bulldozers from Kaggle's Bluebook for Bulldozers
competition — a 400,000-row time series with heavy missing data, taken from raw CSV to
Kaggle-formatted predictions.

Where my [heart disease project](https://github.com/ThaiBenjamin/end-to-end-heart-disease)
was a clean 303-row classification problem, this one is 412,698 rows and 53 columns, with
columns up to 86% missing and dozens of free-text categorical fields. Most of the work is
getting the data into a shape a model will accept at all.

The data splits by time the way the real task does: Train through 2011, Valid covering
January to April 2012, Test covering May to November 2012. Evaluation is RMSLE, to be
minimized.

## Results

Validation set is 2012 sales, 11,573 rows.

| Model | Valid MAE | Valid RMSLE | Valid R² |
|---|---|---|---|
| Baseline (`max_samples=10000`) | $7,177 | 0.2936 | 0.832 |
| RandomizedSearchCV best | $8,063 | 0.3245 | 0.789 |
| Final model (full data, tuned) | $5,951 | 0.2452 | 0.882 |

The final model — `RandomForestRegressor(n_estimators=40, min_samples_leaf=1,
min_samples_split=14, max_features=0.5)` trained on all 401,125 training rows — predicts
auction prices to within about $5,951 on average and explains 88% of the variance in sale
price.

Feature importance is dominated by `YearMade` (20.8%) and `ProductSize` (15.3%), followed
by `fiSecondaryDesc`, `Enclosure`, `fiProductClassDesc`, and `fiBaseModel`. How old the
machine is and how big it is matter more than everything else combined.

## What the notebook covers

**Time series parsing.** Reading the CSV with `parse_dates`, sorting by `saledate`, and
enriching one date column into five features: `saleYear`, `saleMonth`, `saleDay`,
`saleDayOfWeek`, `saleDayOfYear`.

**Missing data.** Filling numeric columns with the median rather than the mean, and adding
an `_is_missing` binary column alongside each so the model can learn from the fact that a
value was absent in the first place.

**Categorical encoding.** Converting string columns to the pandas `category` dtype, then to
numeric codes with `+1` so pandas' `-1` for missing becomes a real category rather than a
sentinel the model misreads.

**A time-based split.** Splitting on `saleYear == 2012` instead of randomly, because
predicting the future from the past is the actual task.

**Evaluation.** Implementing RMSLE from `mean_squared_log_error`, plus a `show_scores()`
helper reporting MAE, RMSLE, and R² for both train and validation.

**Tuning.** `RandomizedSearchCV` over `n_estimators`, `max_depth`, `min_samples_split`,
`min_samples_leaf`, and `max_features` with five-fold CV, using `max_samples=10000` to cut
fit times from minutes to seconds while experimenting.

**Reproducible preprocessing.** Wrapping every transformation into a `preprocess_data()`
function so the test set gets identical treatment, then reconciling column mismatches with
set operations before exporting predictions in Kaggle's `SalesID` / `SalesPrice` format.

## Files

```
end-to-end-bulldozer-price-regression.ipynb   the full analysis
```

The `data/` folder isn't committed — the dataset is around 570 MB and several files exceed
GitHub's 100 MB limit.

## Running it

Download the data from the Kaggle competition page and extract it to
`data/bluebook-for-bulldozers/bluebook-for-bulldozers/`, then:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook end-to-end-bulldozer-price-regression.ipynb
```

## What I took from it

This project was mostly data preparation, and that was the point. My first attempt to fit a
`RandomForestRegressor` failed outright because scikit-learn won't accept strings or NaNs,
which made concrete something that's easy to nod along to in the abstract: models take
numbers, and getting real data to numbers is the actual job.

Two preprocessing ideas stuck. The first is that missingness is itself a signal — adding an
`_is_missing` column before filling with the median means the model can learn that a machine
with no recorded usage hours tends to sell differently, instead of that information being
erased by the fill. The second is that the median is the safe default for numeric fills;
the notebook shows this directly, with a single billion-dollar outlier dragging the mean of
a thousand 100s up to a million while the median doesn't move.

The split taught me the most. Splitting randomly would have let the model train on 2012
sales and then predict 2012 sales, quietly leaking the future into the past and producing a
score that means nothing. Splitting on `saleYear == 2012` mirrors the real task. It's also
why the initial 0.987 R² on the training data was worthless — a random forest can memorize
what it was fit on.

The tuning result was a useful surprise. The `RandomizedSearchCV` model scored *worse* than
the baseline, because the search ran with `max_samples=10000` while the final model trained
on all 401,125 rows. The hyperparameters weren't the bottleneck; the amount of data the
model got to see was. Sometimes the biggest win isn't a better search space, it's letting
the model see everything.

Finally, wrapping preprocessing in a function was not optional. Applying transformations to
the test set by hand produced a column mismatch that only surfaced at prediction time — a
good demonstration of why training and inference should run the same code path.

## Related

- [End-to-End Heart Disease Classification](https://github.com/ThaiBenjamin/end-to-end-heart-disease) — binary classification on clinical patient data
- [End-to-End Dog Vision](https://github.com/ThaiBenjamin/end-to-end-dog-vision) — 120-class image classification with transfer learning
