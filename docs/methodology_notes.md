# Model and Inference Selection for Low-Frequency Continuous Time Observations

Have this in mind whenever you read a paper! The below is your research area, and what you look at should be relevant to it.

## Situation

I have some observations at low frequency of a continuous-time process that may be partially observed and with additive noise. I have T observations of dimension D. This is an example of time series data. Our task is to choose a set of candidate **models** with **hyperparameterisations** and corresponding **inference** procedures to represent this data for the purpose of a particular goal: examples of some of the possible choices for this goal are as follows:

### Goals

- **Explainability**: Being able to explain the underlying dynamics of the process.
- **Prediction/Forecasting** ((At arbitrary times or equidistant times?))
- **Filtering**
- **Smoothing**

Broadly speaking, a model **hyperparameterisation** can make it either fully parametric (FP), semi-parametric (SP) or non-parametric (NP). When using this language, I do not necessarily mean whether the model has parameters/hyperparameters or not: indeed, by that definition a Neural Network would be a parametric model. I am in this context referring to whether the user is using information about the continuous-time process e.g expert knowledge/apriori understanding of the process to influence their modelling choice. The intuition is as follows: if we only have a small dataset and we are expecting to learn the dynamics without giving any other information, then it is likely that we will end up either underfitting if our functional approximator is underparameterised, or overfitting if it is overparameterised.

### Factors

- What is our goal(s) (of the items listed above)
- The length of the data: $T$
- The dimension of the data: $D$
- Are the observations equidistant?
- What is our computational budget?
- How much noise is there in the data/is the data partially observed?
- Is the data generating process a Markov process?
- Do we have information about the dynamics of the data generating process?

If we don't impose any restrictions on the problem, then the possible choices for models out there that we could choose is extremely broad!

Note the following general considerations when thinking about modelling a dependent data problem/time series with a particular choice of **model**, **hyperparameterisation** and **inference** procedure:

- Online or Offline (Inference Procedure)
    - Which is more appropriate could be based on the inferential objective: are we trying to make K-step ahead forecasts, or are we interested in inference on the past/present (nowcasting).
- Modelling Approach: Parametric, Semi Parametric, Non-parametric
    - When deciding this, it is worth taking into consideration what the nature of the underlying process that you are modelling is. If the system under consideration is known to have certain dynamics (e.g Fluids/Particles/Electric Currents), then this could inform the choice of model (ODE/SDE) and the parameterisation.
    - If we want to take a non-parametric approach, under which the dynamics are 'learned' from the data (MACHINE LEARNING IS HOT MAN!) then it is important that you consider what function is being learned, and how informative the data is of the latent function. 
        - Typically, these approaches work by putting in any place where there is a function in a model, which previously needed to be specified by the user, and replacing it with a 'function approximator' like a Neural Net/GP.
- Dataset Size ($T$ length /$D$ state dimension):
    - If we have big data, then we may be able to successfully use non-parametric approaches, such as Neural ODEs, Neural SDEs etc. Furthermore, big data may make some inference methods e,g Particle-based methods/MCMC/Standard GP Regression computationally prohibitive.
    - If we have small data, then it may be harder to use non-parametric approaches, as they involve a large number of parameters. We may need to consider ways to prevent overfitting (e.g regularisation/dropout). 


## Goal

- **Prediction/Forecasting** K-steps ahead - $K << T$

## Factors

- $T$ is small
- $D$ is small
- Observations are not in general equidistant, or we may be interested in forecasting at intermediate points.
- We have enough compute, but not enough to repeatedly implement offline methods.
- Markovian process

## Examples

- Climate Modelling
- Econometrics: - Supply, Demand, Price of a given product.
- Finance: Building a model for the price or volatility of a stock.
- Ecosystems: Modelling predator-prey dynamics.
- Target Tracking: Modelling the motion of a particle that has Newtonian dynamics.

Are there any others?

### Online Inference: Why?

This is a problem of **online** inference. To properly evaluate forecasting performance, it is better to use an **online** modelling approach instead of an offline one, when the data arrives. This is particularly important in the training and model selection stage: evaluating the quality of K-step ahead forecasts if using an offline modelling approach would involve high compute cost from repeated fitting of the model and using a rolling or expanding window to evaluate model performance.

For online inference in this problem, some of the possible modelling approaches that one could take are outlined below, and they warrant looking into in further detail: 

| Model                     | Handles Non-Equidistant Time Points? | Uncertainty Quantification | Online Inference? | Notes                                                   |
|---------------------------|--------------------------------------|----------------------------|--------------------|---------------------------------------------------------|
| Gaussian Process SSM      | Yes                                  | Excellent                 | Yes (with sparse GPs) | Ideal for small data and smooth dynamics.               |
| Neural CDE                | Yes                                  | Good (Bayesian NCDEs)     | Yes                | Best for learning from data, but requires more computation. |
| SDE                       | Yes                                  | Excellent                 | Yes                | Great for noisy systems; drift/diffusion must be specified. |
| Reservoir Computing       | Yes (with preprocessing)             | Good (ensembles)          | Yes                | Efficient and robust; less interpretability.            |
| Bayesian Neural Networks  | Yes                                  | Excellent                 | Yes                | Flexible but computationally intensive for small data.  |
| Particle-Based Methods    | Yes                                  | Excellent                 | Yes                | Computationally intensive but highly flexible.          |
 

The fact that our inference method needs to be Offline Modelling Approaches for continuous time data observed with noise:

There could be an interest in scaleable, online methodology!

I have small data, and therefore it makes sense to want to quantify the uncertainty in my predictions. My aim is to create a model that is most representative of the underlying dynamics of the process, and therefore is able to make good predictions.

We would evaluate the model with:

- K-step ahead forecasts: MSE/MAE across the dataset
- Coverage probabilities for the different models

### Offline: Evaluating the model for K-step ahead prediction

To evaluate the performance of different models, I could take a cross validation approach: split my data into training and test data, fit a bunch of models, then evaluate the performance of my model out of sample for each model, concluding that the model with the lowest prediction error is the best. However, such an approach may result in test dataset overfitting: I may end up picking a model simply because it performs the best on the particular test dataset that I have available, and not because it successfully generalises.   

We can mitigate this issue by having a further 'validation set' - that is unseen until the full modelling exercise is complete. This is used as the final benchmark with which to evaluate model performance. This can only be viewed at the end of the modelling exercise, to ensure that one does not overfit.

To ensure that the models that are fitted do indeed generalise, one can use other metrics of predictive accuracy other than  


- It would be challenging to effectively implement a non-parametric approach: one would struggle to learn the true dynamics of the continuous time process with few data.

  
Say now that I have some observations at low frequency of a continuous-time process with additive noise, I have T observations of dimension D. This is an example of time series data. I assume that both T and D are quite low. Can you come up with a table of the possible modelling approaches that I could use for this dataset. I would like the following methods to be included, but please feel free to add more that you think are relevant:


### Alternative Modelling Approaches

Keep a record here of alternative modelling approaches as you read about them and understand them.