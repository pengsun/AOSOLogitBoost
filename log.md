AOSOLogitBoost
--------
The LogitBoost implementation described in "Sun P., Reid, M.D., Zhou J. AOSO-LogitBoost: Adaptive One-Vs-One LogitBoost for Multi-Class Problems, ICML 2012"


pAOSOLogitBoost
--------
Parallel version (multi-threaded) of AOSOLogitBoost.


pAOSOLogitBoostV2
--------
Improvements to pAOSOLogitBoost:
* Speedup (see matlab/run_script/run_pAOSOLogitBoostV2_xxx.m) of training by 
  ** subsampling instances (Friedman's weight trimming) and 
  ** subsampling features (uniformly sampling at each tree node)
* bugs fixed (crashing due to null leaf)
* redundant computations eliminated