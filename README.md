# Development-of-learning-algorithms-in-the-presence-of-diverse-human-feedback
This is the repository for developing a software framework that aims to integrate multiple forms of feedback and use them in learning algorithms.

Some notes:
- the Expert (or FeedbackModel) knows the environment 
- there are as many type of Experts as many Feedback are available
- there are two levels of abstractions: Trajectory (list of Steps)/Step (pair of state and action); Absolute/Relative (needs two or more steps or trajectories)
- each Expert evaluate the step/trajectory and save it in its history 