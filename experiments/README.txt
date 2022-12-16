These experiments run on the SLURM environment, a common workload manager for HPC clusters. The results are gathered into files in the experiment directory and the WANDB experiment management service (optional).

a) make sure that the code and run scripts are available on the HPC
b) execute the experiments with `run_all.sh` on the HPC
c) after its completed, download and process the results with `prepare_results.sh` (change the SERVERPATH in the script)
d) finally, generate the graphs with `generate_graphs.sh`; optionally, generate the convergence graphs with `graph_debug.sh`
