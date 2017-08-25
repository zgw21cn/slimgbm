from slimgbm.slimgbm.bin import Bin
from slimgbm.slimgbm.histogram import Histogram
__author__ = 'ZhangGuowei'

import numpy as np
import pandas as pd

raw={"f1":np.random.randn(30),'grad':np.random.randn(30),'hess':np.random.randn(30)}
data=pd.DataFrame(raw,columns=['f1','grad','hess'])

hist=Histogram()
hist.construct_bins(data,"f1",5)
hist.find_best_split(1,10)