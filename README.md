# SEAL
This is the code for SEAL.

To run the experiments, you need:

1. Build TVM (we tested on the version published on Feb.21, 2022, we use Python 3.8.10).
2. Build DietCode (in the main branch).
3. Run `nohup python3 solution0.py --cuda 0 > solution0_nohupout0.log 2>&1 &` in TVM environment.
4. Switch to DietCode environment, 
go to the folder `DietCode/tests/codegen`, 
copy the generated schedules to this foler, 
and run `CUDA_VISIBLE_DEVICES=0 python3 run_local_padding_server.py` to measure latency.
