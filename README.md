# Exploratory Data Analysis — Titanic Dataset

A structured EDA project that digs into the Titanic dataset using statistical summaries, correlation analysis, and visualizations to uncover what actually influenced survival.

-----

## Dataset

```
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
```

891 passengers, 12 columns — no manual download needed.

-----

## Setup

```bash
pip install pandas numpy matplotlib seaborn
python eda_analysis.py
```

-----

## What Gets Analyzed

### 1. Dataset Overview

- Row/column count, memory usage
- Per-column: data type, null count, unique values

### 2. Statistical Summary

- Mean, std, min/max, quartiles for all numerical columns

### 3. Missing Values

- Count + percentage missing per column
- Visual heatmap showing where gaps are

### 4. Target Distribution

- How many survived vs died with percentages

### 5. Categorical Breakdowns

- Survival rate broken down by `Pclass`, `Sex`, `Embarked`

### 6. Numerical Comparisons

- Average `Age`, `Fare`, `SibSp`, `Parch` for survivors vs non-survivors

### 7. Correlation Analysis

- Printed ASCII bar chart showing correlation strength/direction with `Survived`
- Full heatmap for all numerical features

### 8. Feature Engineering

|Feature     |Description                                               |
|------------|----------------------------------------------------------|
|`FamilySize`|SibSp + Parch + 1                                         |
|`IsAlone`   |1 if no family aboard                                     |
|`AgeGroup`  |Child / Teen / Adult / Middle-age / Senior                |
|`FareBand`  |Quartile-based fare buckets (Low / Mid / High / Very High)|

-----

## Dashboard — 12 Plots (4×3 grid)

|         |Plot 1                 |Plot 2               |Plot 3                  |
|---------|-----------------------|---------------------|------------------------|
|**Row 1**|Missing values heatmap |Survival count       |Class distribution (pie)|
|**Row 2**|Age distribution       |Fare distribution    |Survival rate by sex    |
|**Row 3**|Survival by class & sex|Age vs Fare scatter  |Survival by age group   |
|**Row 4**|Survival by family size|Survival by fare band|Correlation heatmap     |

Saved as `eda_report.png`.

-----

## Key Findings

- **Women survived at ~74%**, men at ~19% — sex was the strongest predictor
- **1st class passengers** had ~63% survival vs ~24% in 3rd class
- **Children** (age ≤ 12) had a notably higher survival rate
- **Higher fare** strongly correlated with survival — likely because it proxies class
- **Travelling alone** was worse for survival than being with 2–4 family members
- **Pclass** negatively correlates with survival — higher class number = worse odds

-----

## Files

```
├── eda_analysis.py    # main script
├── eda_report.png     # 12-panel dashboard (generated on run)
└── README.md          # this file
```