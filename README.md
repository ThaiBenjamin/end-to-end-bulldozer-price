# 🚜 End-to-End Bulldozer Price Regression

An end-to-end machine learning project — predicting the auction sale price of bulldozers from the Kaggle Bluebook for Bulldozers competition, working through a 400,000-row time series dataset with heavy missing data, from raw CSV to Kaggle-formatted predictions.

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-11557C?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

---

## 📚 What I Was Learning

Regression on a messy, real-world time series dataset — the kind where most of the work is getting the data into a shape a model will accept at all. Where my [heart disease project](https://github.com/ThaiBenjamin/endtoendheartdisease) was a clean 303-row classification problem, this one is 412,698 rows and 53 columns, with columns that are up to 86% missing and dozens of free-text categorical fields.

The project follows the same 6-step framework:

1. **Problem Definition** — *How well can we predict the future sale price of a bulldozer, given its characteristics and previous examples of how much similar bulldozers have sold for?*
2. **Data** — Kaggle's [Bluebook for Bulldozers](https://www.kaggle.com/competitions/bluebook-for-bulldozers/data) competition data, split into Train (through 2011), Valid (Jan–Apr 2012), and Test (May–Nov 2012)
3. **Evaluation** — RMSLE (root mean squared log error) between actual and predicted auction prices, to be minimized
4. **Features** — 53 columns covering machine ID, year made, usage hours, product size, enclosure type, state, and more
5. **Modelling** — `RandomForestRegressor`, tuned with `RandomizedSearchCV`
6. **Experimentation** — Feature importance and what to try next

---

## 🔑 Key Things Practiced

- **Time Series Parsing** — Reading the CSV with `parse_dates`, sorting by `saledate`, and enriching one date column into five features (`saleYear`, `saleMonth`, `saleDay`, `saleDayOfWeek`, `saleDayOfYear`)
- **Handling Missing Data** — Filling numeric columns with the **median** (more robust to outliers than the mean) and adding a `_is_missing` binary column so the model can learn from the fact that a value was absent
- **Categorical Encoding** — Converting string columns to pandas `category` dtype, then to numeric codes with `+1` so pandas' `-1` for missing becomes a real category
- **Time-Based Train/Validation Split** — Splitting on `saleYear == 2012` instead of randomly, because predicting the future from the past is the actual task
- **Custom Evaluation Function** — Implementing RMSLE from `mean_squared_log_error` and a `show_scores()` helper reporting MAE, RMSLE, and R² for both train and validation sets
- **Training on a Subset** — Using `max_samples=10000` to cut fit times from minutes to seconds while experimenting with hyperparameters
- **RandomizedSearchCV** — Searching `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features` over 5-fold CV
- **Reproducible Preprocessing** — Wrapping every transformation into a `preprocess_data()` function so the test set gets identical treatment, then reconciling column mismatches with set operations
- **Kaggle Submission Format** — Exporting predictions as `SalesID` / `SalesPrice`
- **Feature Importance** — Plotting the top 20 features from the trained forest with a `plot_features()` helper

---

## 📊 Results

**Model progression (validation set = 2012 sales, 11,573 rows):**

| Model | Valid MAE | Valid RMSLE | Valid R² |
|-------|-----------|-------------|----------|
| Baseline (`max_samples=10000`) | $7,177 | 0.2936 | 0.832 |
| RandomizedSearchCV best | $8,063 | 0.3245 | 0.789 |
| **Ideal model (full data, tuned)** | **$5,951** | **0.2452** | **0.882** |

The final model — `RandomForestRegressor(n_estimators=40, min_samples_leaf=1, min_samples_split=14, max_features=0.5)` trained on all 401,125 training rows — predicts bulldozer auction prices to within about **$5,951 on average**, explaining **88%** of the variance in sale price.

**Most important features:** `YearMade` (20.8%) and `ProductSize` (15.3%) dominate, followed by `fiSecondaryDesc`, `Enclosure`, `fiProductClassDesc`, and `fiBaseModel` — i.e. how old the machine is and how big it is matter more than everything else combined.

---

## 🗂️ What's Here

```
end-to-end-bulldozer-price-regression.ipynb   # The full analysis
```

The `data/` folder is **not** committed — the dataset is ~570MB and several files exceed GitHub's 100MB file limit.

---

## 🚀 Running It

1. Download the data from the [Kaggle competition page](https://www.kaggle.com/competitions/bluebook-for-bulldozers/data) and extract it to `data/bluebook-for-bulldozers/bluebook-for-bulldozers/`
2. Install the dependencies and launch the notebook:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook end-to-end-bulldozer-price-regression.ipynb
```

---

## 💡 What It Taught Me

This project was mostly data preparation, and that was the point. The first attempt to fit a `RandomForestRegressor` failed outright because scikit-learn won't accept strings or NaNs — which made concrete something that's easy to nod along to in the abstract: models take numbers, and getting real data to numbers is the actual job.

Two preprocessing ideas stuck with me. The first is that **missingness is itself a signal** — adding a `_is_missing` column before filling with the median means the model can learn that, say, a machine with no recorded usage hours tends to sell differently, instead of that information being erased by the fill. The second is that **the median is the safe default** for filling numeric columns; the notebook demonstrates this directly by showing that a single billion-dollar outlier drags the mean of a thousand 100s up to a million while the median doesn't move.

The split taught me the most. Splitting randomly would have let the model train on 2012 sales and then predict 2012 sales — quietly leaking the future into the past and producing a score that means nothing. Splitting on `saleYear == 2012` mirrors the real task. That's also why the initial 0.987 R² on the training data was worthless: a random forest can memorize the data it was fit on.

The tuning result was a useful surprise: the RandomizedSearchCV model scored *worse* than the baseline, because the search ran with `max_samples=10000` while the final model trained on all 401,125 rows. The hyperparameters weren't the bottleneck — the amount of data the model got to see was. Sometimes the biggest win isn't a better search space, it's letting the model see everything.

Finally, wrapping preprocessing in a function was not optional. Applying transformations to the test set by hand produced a column mismatch that only surfaced at prediction time — a good demonstration of why training and inference should run the same code path.
