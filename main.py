"""Run the whole project: optionally re-scrape from scratch, then clean, engineer features,
train the models, analyze depreciation, and draw the figures.

Without a re-scrape, the pipeline starts from the CSVs already in data/raw/.
"""
import shutil

from src import features, figures, preprocess, train
from src.config import HTML_CACHE_DIR
from src.scrapers import craigslist, kbb_msrp, truecar

if __name__ == "__main__":
    answer = input("Scrape all sources from scratch? This deletes the saved pages and re-downloads them. (y = yes, n = no) ")
    steps = [preprocess, features, train, figures]
    if answer.strip().lower() in ("y", "yes"):
        shutil.rmtree(HTML_CACHE_DIR, ignore_errors=True)
        steps = [truecar, craigslist, kbb_msrp] + steps
    for step in steps:
        print(f"\n=== {step.__name__} ===")
        step.main()
