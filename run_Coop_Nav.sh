# example to run CMIX-S on vehicular network routing optimization task
python main_2.py --rl-model cql --mixer qmix --policy-disc --log-dir coop_nav_CMIX-M --application CoopNav --batch-size 32 --training-epochs 100000 --max-env-t 4 --gamma 0.5 --epsilon-scheduler exp --loss-beta 1e-3

