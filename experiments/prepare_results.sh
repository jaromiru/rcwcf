EXP=exp_29_11_2022
datasets=(hepa muta recipes sap_balanced web_100k stats_full toy_b)
SERVERPATH=rci:rcwcf/

mkdir $EXP

echo Syncing...
for d in "${datasets[@]}"; do
	echo 
	echo ---------------------
	echo Syncing dataset $d...
	echo ---------------------
	echo 
	rsync -avz --exclude "wandb" "${SERVERPATH}/${EXP}/${d}_*_rl/" ${EXP}/${d}_rl/ 
	rsync -avz --exclude "wandb" "${SERVERPATH}/${EXP}/${d}_*_mil/" ${EXP}/${d}_mil/
	rsync -avz --exclude "wandb" "${SERVERPATH}/${EXP}/${d}_*_rw/" ${EXP}/${d}_rw/ 
	rsync -avz --exclude "wandb" "${SERVERPATH}/${EXP}/${d}_*_flat/" ${EXP}/${d}_flat/ 
done
wait

echo ====================
echo Gathering results...
ls -d ${EXP}/*_rl | xargs -n 1 -P 4 python3.8 gather_rl.py &
ls -d ${EXP}/*_mil | xargs -n 1 -P 4 python3.8 gather_mil.py &
ls -d ${EXP}/*_rw | xargs -n 1 -P 4 python3.8 gather_rw.py &
ls -d ${EXP}/*_flat | xargs -n 1 -P 4 python3.8 gather_rl.py &
wait
