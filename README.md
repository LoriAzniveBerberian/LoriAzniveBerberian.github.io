**Forecasting California kelp canopy with simple machine learning** <img align="right" width="220" height="220" src="/assets/IMG/image.png">

Hi, I’m Lori and this site hosts my final project for C204-Machine Learning in Atospheric and Oceanic Sciences

In this project, I use a statewide, satellite-derived data set of kelp canopy from the California coast and train a simple machine learning model to predict future kelp canopy from its own history. The goal is test wheather or not I am able to build a **baseline model** that captures typical kelp dynamics without explicitly using environmental drivers. 

---

## links

- [full project report (web)](project.md)  
- main analysis script: [`kelp_ridge_forecast.py`](kelp_ridge_forecast.py?raw=1)
- helper utilities: [`kelp_ml_utils.py`](kelp_ml_utils.py?raw=1)
  
---

## project in one paragraph

Kelp forests are highly dynamic, responding to storms, marine heatwaves, and sediment-laden runoff from land. Many existing kelp models are fairly complex and include multiple environmental variables. Here I ask a simpler question: **how far can we get using only kelp’s own history as a predictor?** I treat each coastal 30 x 30 m pixel as a “station” with a time series of Landsat-derived canopy area, restrict the domain to California, and convert each time series into a supervised learning problem where inputs are the previous four quarters of canopy area and the target is canopy area one quarter ahead. I fit a ridge regression model independently at each station, compare it to a naive persistence baseline, and then map performance along the coast. The model captures strong short-term persistence and an annual “memory” in kelp canopy, and in many regions it explains a substantial fraction of variance beyond a simple copy-the-last-quarter baseline. This makes it a useful starting point and null model for future analyses of how changes in water clarity and light availability impact kelp.

---

## about me

I am a PhD student in Geography at UCLA, studying how changes in water clarity and light availability shape canopy-forming kelps such as giant kelp, bull kelp, split-fan kelp, and sea bamboo, especially under disturbances like wildfire runoff, high river discharge, and marine heatwaves. 
