# Electricity Demand Forecasting: Do ML Models Beat Simple Baselines?

A benchmark of XGBoost against simple seasonal baselines for next-day electricity demand forecasting, with a time-based split, ablations, and error analysis.

## Question
Does a gradient-boosted model with lag features beat simple "same hour in previous days" baselines for hourly electricity demand, and where does it fail?

## Data
- UCI Electricity Load Diagrams 2011-2014 (370 customers, 15-minute readings, 140,256 rows)
- Readings were summed to hourly totals. Only 2012-2014 was used, because many customers were not yet connected in 2011
- 5 customers with no zero-valued hours (a zero can mean "no data"): MT_321, MT_323, MT_324, MT_325, MT_326. They were the first five in column order among customers with complete records, so they are not a random sample
- One model per customer. Errors are in the dataset's own units

## Setup
- **Task:** forecast each hour using only data available at least 24 hours earlier (one day ahead). A 168-hour (one week) horizon was also tested
- **Split by time, never shuffled:**
  - Train: 2012-01-01 to 2014-02-06
  - Validation: 2014-02-06 to 2014-07-20 (3,946 hours)
  - Test: 2014-07-20 to 2014-12-31 (3,946 hours)
- **Metrics:** MAE, RMSE, MAPE (lower is better). Averages are over the 5 customers
- **Baselines:** same hour yesterday; same hour last week; average of the same hour over the previous 7 days
- **XGBoost features (all built from data at least 24 h old):** values 24, 48, and 168 hours earlier; same-hour average over 7 days; mean of the previous 24 hours; hour of day, day of week, month
- **XGBoost settings:** learning rate 0.05, max depth 6, up to 1,000 trees with early stopping on the validation period (it used 77 to 173 trees). No other tuning

## Results (test period, 24 h ahead, mean over 5 customers)

### 1. Baselines
| Method | MAE | RMSE | MAPE % |
|---|---|---|---|
| Average of same hour, last 7 days | 141.78 | 216.63 | 6.35 |
| Same hour yesterday | 150.15 | 248.83 | 6.71 |
| Same hour last week | 181.63 | 267.50 | 8.06 |

### 2. XGBoost vs. best baseline
| Method | MAE | RMSE | MAPE % |
|---|---|---|---|
| Baseline (7-day same-hour average) | 141.78 | 216.63 | 6.35 |
| XGBoost | 132.86 | 203.05 | 6.00 |

MAPE by customer (baseline / XGBoost): MT_321 6.66 / 6.33, MT_323 7.49 / 7.02, MT_324 6.03 / 5.83, MT_325 5.35 / 4.69, MT_326 6.22 / 6.12. XGBoost is better for all five.

### 3. Forecast horizon
| Horizon | Baseline MAE | XGBoost MAE | Baseline MAPE % | XGBoost MAPE % |
|---|---|---|---|---|
| 24 h ahead | 141.78 | 132.86 | 6.35 | 6.00 |
| 168 h ahead | 164.40 | 166.65 | 7.59 | 7.38 |

At 168 h the two are effectively tied (XGBoost is slightly worse on MAE and slightly better on MAPE).

### 4. Ablation (XGBoost, 24 h ahead)
| Removed | MAE | MAPE % |
|---|---|---|
| Nothing | 132.86 | 6.00 |
| Lag features | 143.52 | 6.49 |
| Rolling averages | 135.81 | 6.13 |
| Calendar features (hour, weekday, month) | 137.22 | 6.15 |

### 5. Where the models fail
- Errors are highest around Christmas. Dec 25 has about 79% MAPE for both XGBoost and the baseline, and Dec 24, 27, and 31 are also among the worst days. December averages about 10% MAPE, against 4.4 to 5.6% in the other test months
- Holiday-period days (within one day of a Portuguese public holiday, about 10 days per customer) have about 16% MAPE for all methods, against about 5.4% on other days:

| Method | MAPE, all days | MAPE, normal days | MAPE, holiday period |
|---|---|---|---|
| Baseline (7-day avg) | 6.35 | 5.71 | 16.16 |
| XGBoost | 6.00 | 5.38 | 15.55 |
| XGBoost + holiday flags | 6.02 | 5.38 | 15.87 |

Adding holiday flags did not improve accuracy.

## Findings
- XGBoost with lag features beats the best simple baseline at one day ahead (MAPE 6.00% vs 6.35%, MAE 132.86 vs 141.78) for all five customers. The gain is modest, about 5.5% in MAPE
- The advantage does not hold at one week ahead, where the methods are tied
- Lag features account for most of XGBoost's edge. Without them it does worse than the baseline
- Holidays cause the largest errors for every method, and simple holiday flags did not fix that

## Limitations
- Only 5 customers, chosen by column order. One train/validation/test split and one random seed, so there are no error bars
- The test period (Jul-Dec 2014) contains a single Christmas, so the holiday results rest on very few days
- The ablation, horizon, and error analysis were run on the test period, so they describe behavior and were not used to tune the model
- The holiday calendar assumes the customers are in Portugal. That was not verified
- No ARIMA or neural network (LSTM) baseline was included

## Reproduce
Open `electricity_forecasting.ipynb` in Google Colab and run all cells. The first cell downloads the dataset from UCI.
