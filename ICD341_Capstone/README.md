# Statistical Analysis and Visualization of Maize Production in Zambia Using Python
**name :mainza muya**         
**sudent number :202406802** 

ICD 341 Advanced Statistical Computing: Capstone Project

## Project description
This project analyses district-level maize production in Zambia (10 districts,
2010-2023). It covers data cleaning, descriptive statistics, a regression model
of yield, a bootstrap confidence interval, a Monte Carlo low-yield risk
simulation, and an interactive Streamlit dashboard.

> **Data note:** the dataset in `data/maize_production.csv` is **simulated**
> (fixed seed, see `src/analysis.py::generate_data`) so the project is fully
> reproducible. Results illustrate the methods and are **not** official
> statistics. To use real data (e.g. Zambia Statistics Agency crop forecast
> surveys, FAOSTAT) replace the CSV with one using the same column names:
> `District, Province, Year, Area_ha, Rainfall_mm, Fertilizer_kg_ha, Yield_t_ha, Production_t`.

## Objectives
1. Collect or obtain maize production data.
2. Clean and organise the dataset.
3. Calculate descriptive statistics.
4. Analyse production patterns.
5. Develop appropriate statistical models.
6. Visualise the results.
7. Develop an interactive dashboard.
8. Present findings and recommendations.

## Project structure
```
ICD341_Capstone/
├── data/maize_production.csv       raw (uncleaned) dataset
├── notebooks/maize_analysis.ipynb  full narrative analysis
├── src/analysis.py                 reusable functions + `python src/analysis.py`
├── dashboard/app.py                Streamlit dashboard
├── outputs/graphs.png              summary figure
├── README.md
└── requirements.txt
```

## Requirements
Python 3.9+ then:
```
pip install -r requirements.txt
```

## How to run
1. **Pipeline (prints results, regenerates the figure):** from the project root
   `python src/analysis.py`
2. **Notebook:** `jupyter notebook notebooks/maize_analysis.ipynb`, then Run All Cells.
3. **Dashboard:** from the project root `streamlit run dashboard/app.py`
4. **Export notebook for submission:**
   `jupyter nbconvert --to html notebooks/maize_analysis.ipynb`

## Reproducibility
All random processes use seed `123` (NumPy `default_rng`). Library versions are
listed in `requirements.txt`. No absolute paths are used.

## Author
Student Name | Student Number
