import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv("../../data/train.csv")
profile = ProfileReport(df, title="Pandas Profiling Report")

profile.to_file("../../profile/report.html")
