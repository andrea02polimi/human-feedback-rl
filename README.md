# Development of learning algorithms in the presence of diverse human feedback
This is the repository for developing a software framework that aims to integrate multiple forms of feedback and uses them in learning algorithms. 
The development of the software framework is done as part of my Master Science thesis.

In the first part of the thesis we want to implement a FeedbackModel (or Expert) that given a pair of state and action or a trajectory is able to return a Feedback.

![framework RL + FeedbackModel](img/frameworkRLandFeedbackModel.png)

Dubbi:
- Quando si parla di esperto umano come black box, si assume implicitamente l’esistenza di una funzione deterministica (o stocastica) `F: Trajectory/Step → Feedback`?
- come vogliamo usare questi feedback?

Some notes:
- the Expert is a **black box** entity which can give Feedbacks
- the Expert knows the environment 
- there are as many type of Experts as many Feedback are available
- there are two levels of abstractions: Trajectory (list of Steps)/Step (pair of state and action); Absolute/Relative (needs two or more steps or trajectories)
- each Expert evaluates the step/trajectory and saves it in its history 
- for the preference, until there are not at least two trajectories or two steps there is no feedback 
- for the moment we consider only a passive agent and not an active one that asks the Expert for feedbacks. Therefore no need for "no feedback" as type of Feedback

Some ideas: 
- ~~since the Expert can access the environment it can obtain the reward and give it back to the agent as Feedback.~~
- ~~for preference I may consider the sum of rewards and choose the one that has higher reward~~ 
- ~~correction and demonstration may have a target policy as parameter that deterministically gives the action to be performed given a state (?) but also preference?~~

Implementation ideas:
- ~~FeedbackModel is an abstract class which contains the environment and history attributes. It has three abstract methods required_object_count, mode and scope.~~
- ~~FeedbackModel is inherited by StepFeedbackModel, TrajectoryFeedbackModel, AbsoluteFeedbackModel and RelativeFeedbackModel~~ 
- ~~StepFeedbackModel and TrajectoryFeedbackModel implement the method scope. While AbsoluteFeedbackModel and RelativeFeedbackModel implement quired_object_count and mode~~
- ~~StepFeedbackModel and TrajectoryFeedbackModel introduce an abstract method evaluate which respectively accept a list of (state, action) pair and a list of trajectory~~
- ~~each concrete FeedbackModel checks the scope, mode and implements the evaluate method.~~

- `Expert` is an abstract class which is initialized with the `environment`, the `History` and the `ExpertConfig`
- `ExpertConfig` are predefined configurations 
- `Expert` provides `Feedback` through `query` method

- The Diamond Problem occurs when a class inherits from two classes (like StepFeedbackModel and RelativeFeedbackModel) that both inherit from a common base class (FeedbackModel), leading to ambiguity in the inheritance hierarchy.
To solve this problem we use composition instead of multiple inheritance.

Classes:
- `Expert.py` contains the FeedbackModel 
- `Core.py` contains the core objects like Step, Trajectory and History
- `Feedback.py` contains the Feedback object type returned by the FeedbackModel

Python basic:
- all subclass that inherits from a class that inherits ABC are considered abstract until all the abstract methods are implemented
- for constant value (class attribute) you do not need init, if you have a state (instance attribute) you need the init