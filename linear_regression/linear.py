import pandas as pd
import seaborn as sns
from scipy import stats

nyc = pd.read_csv('ave_hi_nyc_jan_1895-2018.csv')
nyc.head()
nyc.tail()
nyc.columns = ['Date', 'Temperature', 'Anomaly']
nyc.Date = nyc.Date.floordiv(100)
pd.set_option('precision',2)
nyc.Temperature.describe()
linear_regression = stats.linregress(x=nyc.Date, y=nyc.Temperature)
linear_regression.slope
linear_regression.intercept

linear_regression.slope * 2019 + linear_regression.intercept
linear_regression.slope * 1890 + linear_regression.intercept

axes = sns.regplot(x=nyc.Date, y=nyc.Temperature)
axes.set_ylim(10, 70)
