import pandas as pd
import numpy as np
from scipy import stats
import os

'''
Task : Analyze the T-Test problem.
T-Test: there is no difference in math scores between female and male students? (null hypothesis)
'''

def computeTandP(arr1, arr2):
    t, p = 0, 0
    mean1, mean2 = np.mean(arr1), np.mean(arr2)

    std1, std2 = np.std(arr1), np.std(arr2)

    Variance = (((arr1.size-1) * (std1**2)) + ((arr2.size-1) * (std2**2))) / (arr1.size + arr2.size - 2)
    
    sp = np.sqrt(Variance) # standard deviation

    ste = sp * np.sqrt(1/arr1.size + 1/arr2.size) # standard error
    t = (mean1 - mean2) / ste
    df = arr1.size + arr2.size - 2
    
    p = 2 * (1-stats.t.cdf(t, df))

    return t, p

def removeOtliers(arr):
    q3, q1 = np.percentile(arr, [75,25])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    arr = arr[(arr >= lower) & (arr <= upper)]
    return arr

# Read from the scores.csv using pandas
df = pd.read_csv("scores.csv")
# print(df.shape)
# print(df)

fm = df.loc[df['gender'] == 'female', 'math score'].values
mm = df.loc[df['gender'] == 'male', 'math score'].values
# print(type(fm))
# print(fm)

print(f"Size of female data: {fm.size}\nSize of male data: {mm.size}")

fm = removeOtliers(fm)
mm = removeOtliers(mm)

t, p = computeTandP(mm, fm)

print(f"T value: {t} \nP value: {p}") # P value is very small so its reject the null hypothesis

