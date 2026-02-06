# Development of learning algorithms in the presence of diverse human feedback
This is the repository for developing a software framework that aims to integrate multiple forms of feedback and uses them in learning algorithms. 
The development of the software framework is done as part of my Master Science thesis.

In the first part of the thesis we want to project a FeedbackModel that given a pair of state and action or a trajectory is able to return a Feedback.

![framework RL + FeedbackModel](img/frameworkRLandFeedbackModel.png)

Some notes:
- the Expert is a black box entity which can give Feedbacks
- the Expert (or FeedbackModel) knows the environment 
- there are as many type of Experts as many Feedback are available
- there are two levels of abstractions: Trajectory (list of Steps)/Step (pair of state and action); Absolute/Relative (needs two or more steps or trajectories)
- each Expert evaluate the step/trajectory and save it in its history 
- for the preference, until there are not at least two trajectories or two steps there is no feedback 
- for the moment we consider only a passive agent and not an active one that asks the Expert for feedbacks. Therefore no need for "no feedback" as type of Feedback

Some ideas: 
- since the Expert can access the environment it can obtain the reward and give it back to the agent as Feedback.
- for preference I may consider the sum of rewards and choose the one that has higher reward 
- correction and demonstration may have a target policy as parameter that deterministically gives the action to be performed given a state (?) but also preference?