#!/bin/bash

EXP=exp_14_11_2022

python3.8 graph_pareto.py $EXP/toy_b_rl $EXP/toy_b_rw $EXP/toy_b_flat $EXP/toy_b_mil -dest $EXP/toy_b.pdf -styles "k/-/o" "g/--/s" "b/:/X" "r/--/+" -labels "HMIL-CwCF" "RandFeats" "Flat-CwCF" "HMIL" -ylim 0.4,1 &
python3.8 graph_pareto.py $EXP/hepa_rl $EXP/hepa_rw $EXP/hepa_flat $EXP/hepa_mil -dest $EXP/hepa.pdf -styles "k/-/o" "g/--/s" "b/:/X" "r/--/+" -labels "HMIL-CwCF" "RandFeats" "Flat-CwCF" "HMIL" -ylim 0.6,1 &
python3.8 graph_pareto.py $EXP/muta_rl $EXP/muta_rw $EXP/muta_flat $EXP/muta_mil -dest $EXP/muta.pdf -styles "k/-/o" "g/--/s" "b/:/X" "r/--/+" -nolegend -noylabel -labels "HMIL-CwCF" "RandFeats" "Flat-CwCF" "HMIL" -ylim 0.5,1 &
python3.8 graph_pareto.py $EXP/recipes_rl $EXP/recipes_rw $EXP/recipes_flat $EXP/recipes_mil -dest $EXP/recipes.pdf -styles "k/-/o" "g/--/s" "b/:/X" "r/--/+" -nolegend -noylabel -labels "HMIL-CwCF" "RandFeats" "Flat-CwCF" "HMIL" -ylim 0.2,0.8 &
python3.8 graph_pareto.py $EXP/sap_balanced_rl $EXP/sap_balanced_rw $EXP/sap_balanced_flat $EXP/sap_balanced_mil -dest $EXP/sap_balanced.pdf -styles "k/-/o" "g/--/s" "b/:/X" "r/--/+" -nolegend -noylabel -labels "HMIL-CwCF" "RandFeats" "Flat-CwCF" "HMIL" -ylim 0.4,0.8 &
python3.8 graph_pareto.py $EXP/stats_full_rl $EXP/stats_full_rw $EXP/stats_full_flat $EXP/stats_full_mil -dest $EXP/stats_full.pdf -styles "k/-/o" "g/--/s" "b/:/X" "r/--/+" -nolegend -labels "HMIL-CwCF" "RandFeats" "Flat-CwCF" "HMIL" -ylim 0.4,0.6 &
python3.8 graph_pareto.py $EXP/web_100k_rl $EXP/web_100k_rw $EXP/web_100k_flat $EXP/web_100k_mil -dest $EXP/web_100k.pdf -styles "k/-/o" "g/--/s" "b/:/X" "r/--/+" -labels "HMIL-CwCF" "RandFeats" "Flat-CwCF" "HMIL" -ylim 0.6,1 &

wait
