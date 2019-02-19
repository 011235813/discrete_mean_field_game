# Deep Mean Field Games

This is the implementation of all experiments conducted for the ICLR 2018 paper [Learning Deep Mean Field Games for Modeling Large Population Behavior](https://openreview.net/forum?id=HktK4BeCZ)

[ac_irl.py](https://github.com/011235813/discrete_mean_field_game/blob/master/ac_irl.py) is the main code for maximum entropy inverse reinforcement learning and a standard actor-critic RL solver.

[mfg_ac2.py](https://github.com/011235813/discrete_mean_field_game/blob/master/mfg_ac2.py) is an alternative version that implements the same forward RL solver for a pre-specified reward function.

[rlbot_twitter](https://github.com/011235813/rlbot_twitter) (not currently maintained) was used to collect population data for these experiments.
