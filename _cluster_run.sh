#! /bin/sh

# 2 common options. You can leave these alone:-
#$ -j y
#$ -cwd
#$ -m e
#$ -M ckhan.2015@phdis.smu.edu.sg
##$ -q "express.q"
##$ -q "short.q"
#$ -q "long.q"

processorID=$1

source ~/.bashrc
export GRB_LICENSE_FILE=/home/ckhan.2015/gurobi751/gurobi.lic
cd /scratch/ckhan.2015/research/crowdsourcing_logistics

python3 -c "from experiments import init_expEnv; init_expEnv()"
#python3 -c "from experiments import cluster_run; cluster_run($processorID, 20)"