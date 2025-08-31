"This repository contains the code for Gibbs Energy Minimisation.
Currently it is designed for reforming/decomposition of hydrocarbons and 
alcohols in the vapor phase. The code accounts for solid carbon deposition, 
which is of key interest when designing a reactor to minimise coking of 
catalysts." 

The goal of this code and repository is to solve for the mninium gibbs energy of a system of various species, achieved through a minimisation process. This informs us about which specie formation is favoured under a variety of conditions. Currently this code is set up to solve for methanol reforming - $CH_{3}OH+H_{2}O \rightarrow .....$

$$
\text{minimise } \sum_{i}^{N} \Delta G_{f,i}^{O} + R T n_i \ln \frac{p n_i}{p_0 \sum+{\text{gas}} n_i}
$$