import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, norm, poisson
from scipy import stats
from scipy.stats import kurtosis, skew
import seaborn as sns
import pandas as pd

number_cells = 9 # O(n2) ?
number_generations = 1000# O(n)
N = 10000 #resample per cell #(n) 
kenerations = 100

#null model free restriction 
hist=[]
for k in range(kenerations):
    print(k)
    mu = [0] * number_cells
    sigma = [1] * number_cells
    s=[0] * number_cells
    mus=[mu]
    for trial in range(number_generations):
        temp_mu = [0] * number_cells
        temp_sigma = [1] * number_cells 
        for i, mean in enumerate(mu):

            s[i] = np.random.normal(loc=mean,scale=sigma[i],size=N)
            mu_loop = mu[:i] + mu[i+1:]
            for j, meanj in enumerate(mu_loop):
                np.append(s[i],np.random.normal(loc=mean,scale=sigma[i],size=N))
            temp_mu[i] = round(np.mean(s[i]),2)
            temp_sigma[i] = round(np.std(s[i]),2)

        mu = temp_mu
        sigma = temp_sigma
        mus.append(mu)
    mus = pd.DataFrame(mus,columns=range(number_cells),index=range(number_generations+1))
    #s = sns.heatmap(mus,cmap="PiYG");
    #s.set(xlabel='Cells',ylabel='Generations')
    #s;
    #print(sigma[0]/2+sigma[-1]/2)
    hist.append(mus.iloc[number_generations-1,0]-mus.iloc[number_generations-1,8]/kenerations)
print(hist)
plt.hist(hist,bins='auto')
plt.xlabel("Change in mean from first (0th) and last cell (8th)")
plt.ylabel("frequency")
plt.savefig('null_low_restriction.png')
