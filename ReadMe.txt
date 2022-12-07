%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FISEBM (solar.m): ReadMe File
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The 'Floating Ice Sheets Energy Balance Model' (FISEBM) is a coupled ice flow 
and energy balance model based on the papers of Tziperman et al. 2014 and 
Pollard and Kasting (2005).

solar.m is the model source code. For each model iteration (N), the EBM is run
for a specified time (typically 5-10 yrs) and then the ice flow model is run
for a specified time (about 10^6 yrs). The model can be initialized at N=1 or 
from a restart file, where N>1. 

To run solar.m, you will need to change the directory search strings. 
The following directories must be created in the working directory.
Directories: FISRestart, EBMRestart, FISEBMRestart, FISFigures, EBMFigures, 
             Figures, EBMInput
These directories are where the model puts initial conditions, restart files, 
and figures. 

solar.m can be run for a single insolation or in parallel
for an array of insolations. The model is able to simulate 1D ice flow on 
a rapidly-rotating Earth-like planet with a given latitudinal distibution
of insolation. The novelty of this model is its ability to handle gaps
in the sea glaciers with a boundary condition at the leading edge of 
the glacier. See the model documentation for more information.

Francisco Spaulding-Astudillo
University of California, Los Angeles
February 16, 2022
...
The University of Chicago
September 8, 2018

