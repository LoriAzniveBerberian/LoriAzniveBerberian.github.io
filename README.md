**Forecasting California kelp canopy with simple machine learning** <img align="right" width="220" height="220" src="/assets/IMG/image.png">

Hi, I’m Lori and this site hosts my final project for AOS C111 / C204: Environmental Data Science.

In this project, I use a statewide, satellite-derived kelp canopy data set from the California coast and train simple machine learning models to predict future kelp canopy from its own past. The goal is to build a **baseline “persistence-plus” model** that captures typical kelp dynamics without explicitly using environmental drivers. This baseline can later serve as a null model when I look at how changes in water clarity and light (e.g., after wildfires, high runoff years, or marine heatwaves) affect kelp forests.

---

## quick links

- [full project report (web)](project.md)  
- project code (notebook + helpers) – link coming soon

If you are grading this for AOS C111 / C204, the full write-up with figures is in `project.md`, and the executable notebook will be linked above once it is finalized.

---

## project in one paragraph

Kelp forests are highly dynamic, responding to storms, marine heatwaves, and sediment-laden runoff from land. Many existing kelp models are fairly complex and include multiple environmental variables. Here I ask a simpler question: **how far can we get using only kelp’s own history as a predictor?** I treat each coastal 1 km pixel as a “station” with a time series of Landsat-derived canopy area, restrict the domain to California, and convert each time series into a supervised learning problem where inputs are the previous four quarters of canopy and the target is canopy one quarter ahead. I fit a ridge regression model independently at each station, compare it to a naive persistence baseline, and then map performance along the coast. The model captures strong short-term persistence and an annual “memory” in kelp canopy, and in many regions it explains a substantial fraction of variance beyond a simple copy-the-last-quarter baseline. This makes it a useful starting point and null model for future analyses of how changes in water clarity and light availability impact kelp.

---

## about me

I am a PhD student in Geography at UCLA, studying how changes in water clarity and light availability shape shallow benthic habitats and canopy-forming kelps such as giant kelp, bull kelp, split-fan kelp, and sea bamboo, especially under disturbances like wildfire runoff, high river discharge, and marine heatwaves. This class project is a small, contained experiment that connects directly to my dissertation: building quantitative baselines for kelp canopy dynamics so that future disturbances can be interpreted relative to “business-as-usual” variability in light and habitat conditions.
