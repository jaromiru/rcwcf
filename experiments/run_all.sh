EXP=exp_29_11_2022
export WANDB_RUN_GROUP=$EXP
mkdir -p $EXP
ln -s ../data ${EXP}/data

# NOTES:
# cpulong=72hrs; cpufast=4hrs
# datasets: toy=synthetic; web=threatcrowd; recp=ingredients

# HMILwCF
sbatch --exclude=n33 -J muta_rl --mem=4GB exp_rl.sh $EXP muta 1000 200000 256 0.025 0.00025 1e-4
sbatch --exclude=n33 -J hepa_rl --mem=4GB exp_rl.sh $EXP hepa 1000 200000 256 0.1 0.005 1e-4
sbatch --exclude=n33 -J recp_rl --mem=4GB exp_rl.sh $EXP recipes 1000 300000 256 0.05 0.0025 0.3 
sbatch --exclude=n33 -J sap_rl --mem=4GB exp_rl.sh $EXP sap_balanced 1000 200000 256 0.1 0.005 1e-4
sbatch --exclude=n33 -J stats_rl --mem=4GB exp_rl.sh $EXP stats_full 1000 200000 256 0.05 0.0025 1e-4
sbatch --exclude=n33 -J web_rl --mem=4GB exp_rl.sh $EXP web_100k 1000 200000 256 0.05 0.0025 3.0
sbatch --exclude=n33 -J toy_rl -p cpufast --mem=4GB exp_rl.sh $EXP toy_b 100 20000 256 0.025 0.00025 1e-4

# MIL
sbatch --exclude=n33 -J muta_mil --mem=4GB exp_mil.sh $EXP muta 100 5000 256 1.0
sbatch --exclude=n33 -J hepa_mil --mem=4GB exp_mil.sh $EXP hepa 100 5000 256 1.0
sbatch --exclude=n33 -J recp_mil --mem=4GB exp_mil.sh $EXP recipes 100 20000 256 1.0 
sbatch --exclude=n33 -J sap_mil --mem=4GB exp_mil.sh $EXP sap_balanced 100 5000 256 0.1
sbatch --exclude=n33 -J stats_mil --mem=4GB exp_mil.sh $EXP stats_full 100 5000 256 3.0
sbatch --exclude=n33 -J web_mil --mem=4GB exp_mil.sh $EXP web_100k 100 5000 256 3.0
sbatch --exclude=n33 -J toy_mil --mem=4GB exp_mil.sh $EXP toy_b 100 2000 256 1.0

# Random Walk
sbatch --exclude=n33 -J muta_rw --mem=4GB exp_rw_10.sh $EXP muta 1000 20000 256 1.0
sbatch --exclude=n33 -J hepa_rw --mem=4GB exp_rw_40.sh $EXP hepa 1000 20000 256 1e-4
sbatch --exclude=n33 -J recp_rw --mem=4GB exp_rw_10.sh $EXP recipes 1000 100000 256 1.0 
sbatch --exclude=n33 -J sap_rw --mem=4GB exp_rw_10.sh $EXP sap_balanced 1000 40000 256 0.1
sbatch --exclude=n33 -J stats_rw --mem=4GB exp_rw_10.sh $EXP stats_full 1000 40000 256 3.0
sbatch --exclude=n33 -J web_rw --mem=4GB -p cpulong --time=72:00:00 exp_rw_10.sh $EXP web_100k 1000 20000 256 3.0
sbatch --exclude=n33 -J toy_rw -p cpufast --time=04:00:00 --mem=4GB exp_rw_20.sh $EXP toy_b 100 2000 256 1e-4

# Flat
sbatch --exclude=n33 -J muta_flat --mem=4GB exp_flat.sh $EXP muta 1000 200000 256 0.05 0.0025 1.0
sbatch --exclude=n33 -J hepa_flat --mem=4GB exp_flat.sh $EXP hepa 1000 200000 256 0.1 0.005 1.0
sbatch --exclude=n33 -J recp_flat --mem=4GB exp_flat.sh $EXP recipes 1000 200000 256 0.05 0.0025 1.0
sbatch --exclude=n33 -J sap_flat --mem=4GB exp_flat.sh $EXP sap_balanced 1000 200000 256 0.05 0.0025 0.1
sbatch --exclude=n33 -J stats_flat --mem=4GB exp_flat.sh $EXP stats_full 1000 200000 256 0.05 0.0025 1.0
sbatch --exclude=n33 -J web_flat --mem=4GB exp_flat.sh $EXP web_100k 1000 200000 256 0.05 0.0025 3.0
sbatch --exclude=n33 -J toy_flat -p cpufast --time=04:00:00 --mem=4GB exp_flat.sh $EXP toy_b 100 20000 256 0.05 0.0025 1e-4