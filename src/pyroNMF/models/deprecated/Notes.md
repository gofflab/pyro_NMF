- Hierarchical bayes
    - Normally distributed data is decomposed into two matrices that are gamma distributed
    - Gammas are the sum of exponentials and each element in A and P matrix is the sum of the weights of its atoms
    - Exponential distributions have the majority of their mass at zero
    - The weights of the atoms are gamma distributed
    - For each gene in each sample (plate is genes + sample)
        - the amplitued (A_mean) is the sum of the weights of the atoms assigned to that cell in the matrix
        - cell becomes Gamma distributed because it's the sum of exponentials
    - Question: Does sampling each cell (genexsample) from a gamma distribution abrogate the need for the atoms?
    - Number of atoms is poisson
    - How are you constraining the gammas to have the most weight in each cell with the fewest atoms?
    - This is what the prior distribution on the mean (a) of the gamma is doing.
        That prior is the combined exponentials with the number of exponentials being poisson distributed.
GAPS-JAGS
- JAGS is a program for Bayesian analysis of complex statistical models using Markov Chain Monte Carlo (MCMC) methods- 
- ELANA is using the underlying distributional assumptions of the model to infer the parameters in GAPS-JAGS
