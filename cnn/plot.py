#!/usr/bin/env python

import re
import os
import sys
import json
import pandas as pd
import numpy as np

from rsub import *
from matplotlib import pyplot as plt

def smart_json_loads(x):
    try:
        return json.loads(x)
    except:
        pass

all_data = []
for p in sys.argv[1:]:
    acc = list(map(float, open(p).read().splitlines()))
    acc = np.array(acc) / 100
    _ = plt.plot(acc, alpha=0.75, label=re.sub('/', '__', p))

_ = plt.grid(alpha=0.25)
for t in np.arange(0.90, 1.0, 0.01):
    _ = plt.axhline(t, c='grey', alpha=0.25, lw=1)

_ = plt.legend()
_ = plt.ylim(0.94, 1.0)
show_plot()
