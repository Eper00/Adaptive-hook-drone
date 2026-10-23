#!/usr/bin/env bash
# Seed study of the thesis (Fig. 3.1 and Section 3.3.8): 10M steps, seeds 1-3,
# all runs in parallel (one CPU thread each).
#   <out>/transport_s<k>             flat agent without curriculum
#   <out>/transport_s<k>_curriculum  flat agent with curriculum
#   <out>/director_s<k>_curriculum   high-level (director) agent with curriculum
# Logs: <out>/logs/. Plot: python -m utilities.plot_seeds --root <out>
#
# Usage: bash utilities/train_seeds.sh [timesteps] [all|cl|nocl] [ent_coef] [out]
#   cl:   only the curriculum runs (transport + director), nocl: only transport without CL
#   defaults: 10000000 all 0.01 results/seeds (the setup of the thesis)
# e.g.  bash utilities/train_seeds.sh 10000000 all 0.005 results/seeds_ent0.005   (other entropy coefficient)
set -e
cd "$(dirname "$0")/.."
source venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
STEPS=${1:-10000000}
WHICH=${2:-all}
ENT=${3:-0.01}
OUT=${4:-results/seeds}
mkdir -p $OUT/logs
for seed in 1 2 3; do
  if [ "$WHICH" != cl ]; then
    nohup python utilities/learn.py --env_type adaptive_transport --timesteps $STEPS --curriculum_flag false \
      --seed $seed --ent_coef $ENT --output_dir $OUT/transport_s$seed > $OUT/logs/transport_s$seed.log 2>&1 &
  fi
  if [ "$WHICH" != nocl ]; then
    nohup python utilities/learn.py --env_type adaptive_transport --timesteps $STEPS --curriculum_flag true \
      --seed $seed --ent_coef $ENT --output_dir $OUT/transport_s$seed > $OUT/logs/transport_s${seed}_curriculum.log 2>&1 &
    nohup python utilities/learn.py --env_type adaptive_director --timesteps $STEPS --curriculum_flag true \
      --seed $seed --ent_coef $ENT --output_dir $OUT/director_s$seed > $OUT/logs/director_s${seed}_curriculum.log 2>&1 &
  fi
done
echo "started ($WHICH, ent_coef $ENT), logs in $OUT/logs"
