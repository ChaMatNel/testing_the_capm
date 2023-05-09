# Testing the CAPM
This repository contains the code that I wrote to test the Capital Asset Pricing Model.

## Process
I collected data on 100 randomly chosen stocks over a five-year period, along with the risk-free rate and the returns on the market portfolio. I then ran 2 seperate regressions. In the first regression I estimated the Beta for each stock. In the second regression I estimated the risk premium to test the null hypothesis that the beta of a stock is the only significant predictor of a stock's risk premium.

## Results
My resluts rejected the CAPM. The second pass regression shows the coefficients on the variance of the residuals and intercept to be statistically different from 0. An F-test that γ_1= -0.0041 reported an F-value of 0.0027 which means that the stock’s beta coefficient is statistically different from -0.0041 (the average risk premium).
