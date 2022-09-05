# SEAL
This is the code for SEAL.

To run the experiments, you need:

1. Build TVM (we tested on the version published on Feb.21, 2022, we use Python 3.8.10).
2. Build DietCode (in the main branch).
3. Run `nohup python3 solution0.py --cuda 0 > solution0_nohupout0.log 2>&1 &` in TVM environment.
4. Run the generated schedules in DietCode environment to measure latency.
