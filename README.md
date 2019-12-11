# Gridworld-Passive-BCI-Experiment
This is a collection of files to train agents in a gridworld environment and run a passive brain-computer interface (BCI) experiment.  In the experiment, a hunter will attempt to catch the prey.  Errors were purposely introduced so that the human observer can evaluate each of the hunter's actions.  In parallel, optimal policy agents observe the same experiment and generate prediction error in order to correlate this with the human's event-related potential magnitude.

Module Requirements:\
numpy, pickle, pygame, matplotlib, seaborn

Training Agents:\
Run the file "train_agents.py"

Experiment:\
Run the file "run_experiment_vis.py" for a visual experiment\
Run the file "run_experiment.py" for a non-visual experiment to test accuracy of agent/s

Observation Prediction Error Generation:\
Run the file "run_agents_observation.py"

Note: To conduct experiment with EEG, a port system should be set up with trigger codes relaying information at the hunter's step.

# Abstract
Typical passive brain-computer interface (BCI) paradigms detect whether or not an error is present in the EEG signal in order to update the paradigm to reflect the user’s intentions. This update, however, is binary in nature and does not necessarily reflect the human’s true policy. In our study we demonstrate a method to understand the magnitude of the event-related potential (ERP) in both error and non-error trials. We used simulation prediction error generated from optimal policy agents observing a gridworld experiment. This prediction error is then mapped to the ERP in order to better understand the human’s policy. Our results indicate a partial correlation at
different time regions within the ERP, with the strongest positive correlation at the rebound spike following the initial drop in ERP magnitude. Full correlations across all trials were weak potentially due to the adaptive nature of the human policy not being well represented by static simulation agents. Further studies are required in order to determine a better relationship between ERP magnitude and simulation prediction error to create a closed loop BCI paradigm in which the updates are more reflective of the human’s natural policy.
