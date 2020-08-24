import math
import scipy as sc
from scipy import stats
import pandas as pd
import numpy as np
def cramers_corrected_stat(x,y):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    
    confusion_matrix =pd.crosstab(x,y)
    chi2 = sc.stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
#     print(r)
#     print(k)
#     print(phi2)
    phi2corr = max([0]+(phi2 - ((k-1)*(r-1))/(n-1)))   
#     print(phi2corr)
    rcorr = r - (((r-1)**2)/(n-1))
    kcorr = k - (((k-1)**2)/(n-1))
    
#     print(kcorr)
#     print(rcorr)
#     print(n)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

import collections as coll
def conditional_entropy(x,y):
    # entropy of x given y
    y_counter = coll.Counter(y)
    xy_counter = coll.Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        if p_y==0:
            entropy +=0
        else:
            entropy += p_xy * math.log(p_y/p_xy)
    return entropy




def theils_u(x, y):
    """ calculate Thiels_u conditional information statistic for categorial-categorial directed association.
    
    """
    s_xy = conditional_entropy(x,y)
    x_counter = coll.Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = sc.stats.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x
