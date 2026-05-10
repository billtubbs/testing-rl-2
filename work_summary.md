Summary by Claude of recent work

  Session 1 — May 3 (ba6b6d65)                                                                                                                              
  - Discussed common RL libraries (RLlib, SB3, etc.)                                                                                                        
  - Set up the project with Stable Baselines3: PPO, SAC, A2C, TD3                                                                                         
  - Decided on the CartPole-BT-vL-v1 custom gym environment                                                                                                 
  - Recreated the conda environment; discussed PILCO and MBPO (model-based, deferred)                                                                       
                                                                                                                                                            
  Session 2 — May 5 (e3491ce7)                                                                                                                              
  - Initial PPO (1M ts) was done; SAC still running — noted those runs lacked explicit seeds (not reproducible)                                             
  - Built a generic train.py script with CLI args and per-seed subdirectories                                                                               
  - Discussed adding A2C; asked about a 4th algorithm (TD3 was the answer)                                                                                  
                                                                                                                                                            
  Session 3 — May 6 (548bced7)                                                                                                                              
  - Added a --resume option to train.py to continue training from a checkpoint                                                                              
  - Discussed renaming trained model files to preserve results during resume                                                                                
  - Refactored the Snakemake file to specify individual experiments instead of all combos of algos/envs/seeds                                               
  - Changed directory structure to use experiment name (e.g. dL-vL-tqc/seed_0/150000ts.zip)                                                                 
                                                                                                                                                            
  Session 4 — May 6 (fe505660)                                                                                                                              
  - Pathlib question (listing files as strings)                                                                                                             
  - Added total timesteps to eval_all.py output                                                                                                             
  - Fixed verbose log messages from SB3 evaluation
  - Debugged a column naming issue (eval=106 → should be seed)                                                                                              
  
  Session 5 — May 7 (6feeaaa7)
  - Worked on analyse-results.ipynb: horizontal bar plot comparing RL algorithms vs LQR/LQG controller                                                      
  - Reorganized into a 3×2 subplot layout (same env with/without measurement noise per row)                                                                 
  - Fixed deprecation warnings in the best-seed lookup code                                
  - Simplified y-label placement using nested loops                                                                                                         

  Session 6 - May 9
  - Wrote a script to compare different reward functions that treat theta as a circular variable
