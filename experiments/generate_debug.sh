EXP=exp_14_11_2022
datasets=(hepa muta recipes sap_balanced web_100k stats_full toy_b)

for d in "${datasets[@]}"; do
	echo $d
	(cd ${EXP}; ls -1 ${d}_rl/*_29.log) | sed 's/\.log//' | xargs -n1 -P4 -i python3.8 graph_debug.py "${EXP}/{}.log" -title "{}" -dest "${EXP}_debug" &
	(cd ${EXP}; ls -1 ${d}_rl/*_20.log) | sed 's/\.log//' | xargs -n1 -P4 -i python3.8 graph_debug.py "${EXP}/{}.log" -title "{}" -dest "${EXP}_debug" &
	(cd ${EXP}; ls -1 ${d}_rl/*_10.log) | sed 's/\.log//' | xargs -n1 -P4 -i python3.8 graph_debug.py "${EXP}/{}.log" -title "{}" -dest "${EXP}_debug" &
	(cd ${EXP}; ls -1 ${d}_rl/*_1.log) | sed 's/\.log//' | xargs -n1 -P4 -i python3.8 graph_debug.py "${EXP}/{}.log" -title "{}" -dest "${EXP}_debug" &
	wait
done
