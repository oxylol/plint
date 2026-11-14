"""Stocks: stock-trading models on paper, side by side."""
import warnings

# A day with no stocks, or a stock with no bars in a window, is an ordinary NaN here, not news.
for _msg in ("Mean of empty slice", "All-NaN slice encountered", "Degrees of freedom <= 0", "invalid value encountered"):
    warnings.filterwarnings("ignore", message=_msg, category=RuntimeWarning)
