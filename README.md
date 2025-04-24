# Replication of Models, Regimes, and Trend Following (Parts 1 & 2)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/internQuant/models-regimes-trend/blob/main/notebooks/Parts_1%262.ipynb)

This repo contains a Jupyter notebook that steps through an educational replication of JungleRock’s white papers “Models, Regimes, and Trend Following – Parts 1 & 2.” Results closely match the originals, with minor discrepancies noted.

---

## ▶️ Running the Notebook

### 🔄 Option 1: Google Colab (Recommended)

You can run the notebook directly in Colab, no installation needed:

1. Go to the repo [Colab](https://colab.research.google.com/github/internQuant/models-regimes-trend/blob/main/notebooks/Parts_1%262.ipynb).
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

Clone the repository and then launch the notebook with Jupyter and open `Parts_1&2.ipynb`.

---

## 🧭 Notebook Overview

1. **Setup & Imports**  
   Load libraries and helper modules.

2. **Data Ingestion**  
   Download daily & monthly Market – RF returns directly from professor Kenneth French's data library via `FFScraper`.

3. **Strategy Implementations**  
   - Simple trend-following  
   - Volatility-managed sizing  
   - Regime-based filters

4. **Backtests & Metrics**  
   Compare cumulative returns, Sharpe ratios, drawdowns, turnover.

5. **Visualizations**  
   Reproduce charts from the white papers.

6. **Results Comparison**  
   Highlight differences vs. published numbers and discuss causes.

---

## 📑 Acknowledgements

Special thanks to the [**JungleRock**](https://junglerock.com/) team for providing the white papers this work attempts to replicate.

---

## 📖 References

- JungleRock, “Models, Regimes, and Trend Following – Part 1”  
- JungleRock, “Models, Regimes, and Trend Following – Part 2”  
- Fama, E. F. & French, K. R. (1993). “Common risk factors in the returns on stocks and bonds.”

## 📖 Technical References

- StataCorp. (2015). *MSWITCH: Markov-switching regression models*. [Stata Manual](https://www.stata.com/manuals14/tsmswitch.pdf)
- Hamilton, J. D. (1994). *Time Series Analysis*, Chapter 22. Princeton: Princeton University Press.

---
### **Parts 3 & 4 are in the works and will be added soon. Stay tuned!**