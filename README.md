# Replication of Models, Regimes, and Trend Following (Parts 1 & 2)

[![Open In Colab (Parts 1 & 2)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/internQuant/models-regimes-trend/blob/main/notebooks/Parts_1%262.ipynb) **Parts 1&2**  

[![Open In Colab (Part 3)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/internQuant/models-regimes-trend/blob/main/notebooks/Part_3.ipynb) **Part 3**

[![Open In Colab (Part 3)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/internQuant/models-regimes-trend/blob/main/notebooks/Part_4.ipynb) **Part 4**

>This repository contains Jupyter notebooks that step through an educational replication of JungleRock’s white paper series “Models, Regimes, and Trend Following" (all parts).

> Results closely match the originals, discrepancies are noted and commented.

---

## ▶️ Running the Notebooks

### 🔄 Option 1: Google Colab (Recommended)

You can run the notebooks directly in Colab, no installation needed:

1. Go to the repo [Parts 1 & 2 Colab](https://colab.research.google.com/github/internQuant/models-regimes-trend/blob/main/notebooks/Parts_1%262.ipynb), [Part 3 Colab](https://colab.research.google.com/github/internQuant/models-regimes-trend/blob/main/notebooks/Part_3.ipynb) or [Part 4 Colab](https://colab.research.google.com/github/internQuant/models-regimes-trend/blob/main/notebooks/Part_3.ipynb).
2. Login if needed.
3. Run cells.

---

### 💻 Option 2: Local Setup

To run locally, make sure you have Python 3.8+ and the following libraries installed:

- `pandas`
- `numpy`
- `matplotlib`
- `numba`
- `requests`
- `statsmodels`
- `cycler`
- `scikit-learn`

Clone the repository and then launch the desired notebook with Jupyter.

---

## 🧭 Notebooks Overview
### Notebooks run through systematic steps:

1. **Setup & Imports**  
   Load libraries and helper modules.

2. **Data Ingestion**  
   Download daily & monthly Market – RF returns directly from professor Kenneth French's data library via `FFScraper` or directly load via included csv files.

3. **Strategy Implementations**  
   Replicate transformations done in the orginal work

4. **Backtests & Metrics**  
   Compare cumulative returns, Sharpe ratios, drawdowns, turnover.

5. **Visualizations**  
   Reproduce charts.

6. **Results Comparison**  
   Highlight differences vs. published numbers and discuss possible causes.

---

## 📑 Acknowledgements

Special thanks to the [**JungleRock**](https://junglerock.com/) team for providing the white papers this work attempts to replicate.

---

## 📖 References

- JungleRock, “Models, Regimes, and Trend Following – Part 1”  
- JungleRock, “Models, Regimes, and Trend Following – Part 2”  
- JungleRock, “Models, Regimes, and Trend Following – Part 3” 
- JungleRock, “Models, Regimes, and Trend Following – Part 4” 
- Fama, E. F. & French, K. R. (1993). “Common risk factors in the returns on stocks and bonds.”

## 📖 Technical References

- StataCorp. (2015). *MSWITCH: Markov-switching regression models*. [Stata Manual](https://www.stata.com/manuals14/tsmswitch.pdf)
- Hamilton, J. D. (1994). *Time Series Analysis*, Chapter 22. Princeton: Princeton University Press.

---
