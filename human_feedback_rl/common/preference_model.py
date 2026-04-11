from .core import Preference, SegmentPair


class PreferenceModelFromReward:
    """
    Synthetic preference oracle based on the true environment reward.

    Given two segments, assigns label=1.0 if the first has higher cumulative
    true reward, label=0.0 if the second does, and label=0.5 for ties.
    Used to generate supervised training signal for the reward model without
    requiring actual human feedback.
    """

    def __call__(self, pair: SegmentPair) -> Preference:
        r1 = pair.seg1.true_return
        r2 = pair.seg2.true_return
        if r1 > r2:
            label = 1.0
        elif r2 > r1:
            label = 0.0
        else:
            label = 0.5
        return Preference(seg1=pair.seg1, seg2=pair.seg2, label=label)