# A-hybrid-compartmental-model-with-a-case-study-of-COVID19-in-Great-Britain-and-Israel
The scripts accompanying the paper "A Hybrid Compartmental Model with a Case Study of COVID-19 in Great Britain and Israel" by Malaspina et al., available at https://arxiv.org/abs/2202.01198

  seird.py          - a script that defines the SEIRD model described in the paper.
  help_functions.py - a script with functions for preprocessing the data and creating the vector of (temporal) exposure probabilities.
  netwroks.py       - a script that contains a function for creating the underlying graphs of the modeled population.
  Model.py          - a script that combines the above functions to give a complete model as proposed in the paper.
  run_ISR.py        - a script with the optimal parameters estimated for Israel, run over an entire timeframe to give the numerical results.
  run_GBR.py        - a script with the optimal parameters estimated for Great Britain, run over an entire timeframe to give the numerical results.
