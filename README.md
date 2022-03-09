# DSC180B-A11-Project
## Particle Jet Mass Regression

<img src='https://raw.githubusercontent.com/isacmlee/particle-physics-visuals/main/images/cern_atlas.jpeg'>


With jet data collected from proton-proton collision, the contents in this repository aim to predict the mass of particle jet, which is an important feature in classifying the types of those jets.
The model is implemented based on PyTorch Neural Network APIs and gets fitted to training sets available on DSMLP server.

#### Repository Structure:

 - `conf`: directory in which all necessary variables needed for configuration in development are stored
 - `data`: temporary directory that explains the source of our data
 - `notebooks`: directory in which .ipynb notebooks for EDA and model evaluation is stored
 - `src`: source directory for all library codes used for this project
 - `run.py`: main script to run within properly set-up environment using `python3 run.py train` for model training or `python3 run.py test` for model assessment
 - `simplenetwork_best.pt`: PyTorch-based file that stores weights of optimized, or best "fitting," model
