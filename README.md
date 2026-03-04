# Human Feedback RL
This is the repository for developing a software framework that aims to integrate multiple forms of feedback and uses them in **Learning Algorithms**. 

*The development of the software framework is done as part of my Master Science thesis.*

In the first part of the thesis the objective is to implement a FeedbackModel (or Expert) that given a pair of state and action (i.e. one `Step`) or a `Trajectory` (a sequence of Steps) is able to return a `Feedback`.

![framework RL + FeedbackModel](img/frameworkRLandFeedbackModel.png)

There are four different types of `Feedback`: 
- `CorrectionFeedback`
- `DemonstrationFeedback`
- `RewardFeedback`
- `PreferenceFeedback`

For each type of `Feedback` we have an `Expert` which provides feedback per `Step` and per `Trajectory` (continuously), and these are all **abstract classes** to be implemented for a specific use case, for this thesis the objective is to test the framework in simulated autonomous driving (SUMO):
- `SpepCorrectionExpert`, `TrajectoryCorrectionExpert`
- `SpepDemonstrationExpert`, `TrajectoryDemonstrationExpert`
- `SpepRewardExpert`, `TrajectoryRewardExpert`
- `SpepPreferenceExpert`, `TrajectoryPreferenceExpert`