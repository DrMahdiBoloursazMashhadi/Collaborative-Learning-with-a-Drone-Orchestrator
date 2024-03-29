# Collaborative-Learning-with-a-Drone-Orchestrator
This repo accompanies submission: M. B. Mashhadi, M. Mahdavimoghadam, R. Tafazolli and W. Saad, "Collaborative Learning with a Drone Orchestrator," in IEEE Transactions on Vehicular Technology, doi: 10.1109/TVT.2023.3303630.

In this research, the problem of drone-assisted collaborative learning is considered. In this scenario, swarm of intelligent wireless devices train a shared neural network (NN) model with the help of a drone. Using its sensors, each device records samples from its environment to gather a local dataset for training. The training data is severely heterogeneous as various devices have different amount of data and sensor noise level. The intelligent devices iteratively train the NN on their local datasets and exchange the model parameters with the drone for aggregation. For this system, the convergence rate of collaborative learning is derived while considering data heterogeneity, sensor noise levels, and communication errors, then, the drone trajectory that maximizes the final accuracy of the trained NN is obtained. The proposed trajectory optimization approach is aware of both the devices data characteristics (i.e., local dataset size and noise level) and their wireless channel conditions, and significantly improves the convergence rate and final accuracy in comparison with baselines that only consider data characteristics or channel conditions. Compared to state-of-the-art baselines, the proposed approach achieves an average 3.85% and 3.54% improvement in the final accuracy of the trained NN on benchmark datasets for image recognition and semantic segmentation tasks, respectively. Moreover, the proposed framework achieves a significant speedup in training, leading to an average 24% and 87% saving in the drone’s hovering time, communication overhead, and battery usage, respectively for these tasks.

![system-model5](https://github.com/DrMahdiBoloursazMashhadi/Collaborative-Learning-with-a-Drone-Orchestrator/assets/121172212/3bbc5dbf-116c-4a62-90a9-298ae4581ea7)

## Usage
The MATLAB codes in folder "Trajectory Optimization" are provided to optimize the trajectory points for both AIoT and autonomous vehicles scenarios. Simply set the parameters of the intended simulation scenario and run "MainTrajectoryOptimization.m". This will optimize the trajectory points and output the resulting packet error rates in "errors.mat". Move the "errors.mat" file to the folder of interest and run the provided Python codes for the AIoT and autonomous vehicles to train the models and generate the learning curves. Do not forget to change all directory addresses as necessary.

## Citation
If using this repo please cite: 
[*] M. B. Mashhadi, M. Mahdavimoghadam, R. Tafazolli and W. Saad, "Collaborative Learning with a Drone Orchestrator," in IEEE Transactions on Vehicular Technology, doi: 10.1109/TVT.2023.3303630.

## Questions?
For any questions related to this repo, feel free to contact me at m.boloursazmashhadi@surrey.ac.uk or raise an issue within this repo. I will do my best to reply as soon as possible.
