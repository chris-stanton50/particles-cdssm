# FHN Smoothing via iCSMC

Re-run the FHN smoothing experiment for the paper "Particle-based inference for continuous-discrete state space models" here.

The details of the experiment are outlined in Secion 6.2.2 of the paper. To run the experiment, run the python script with command:

`python fhn_smoothing.py 100`

This will take a few hours. The results will then be stored at `./results`.
The results plots can then be reproduced in the notebook 

`fhn_smoothing_results.ipynb`. 

From running this notebook, the plots of the will be saved and stored in: 
 
 - `./figures/fig_6_fhn_results.pdf`
 - `./figures/fig_7_r_hat_results.pdf`
