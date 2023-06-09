from federated_learning.utils import replace_0_with_2
from federated_learning.utils import replace_5_with_3
from federated_learning.utils import replace_1_with_9
from federated_learning.utils import replace_4_with_6
from federated_learning.utils import replace_1_with_3
from federated_learning.utils import replace_6_with_0
from federated_learning.worker_selection import RandomSelectionStrategy
from server import run_exp

if __name__ == '__main__':
    START_EXP_IDX = 4000
    NUM_EXP = 1 # for now, just run one experiment
    # NUM_POISONED_WORKERS = 1
    # REPLACEMENT_METHOD = replace_1_with_9 # don't use it
    KWARGS = {
        "NUM_WORKERS_PER_ROUND" : 50
        # "NUM_WORKERS_PER_ROUND" : 10 # exp11, exp12_poison_all_every_round_1000_comm_rounds
        # "NUM_WORKERS_PER_ROUND" : 1
        # "NUM_WORKERS_PER_ROUND" : 2
    }

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_exp(KWARGS, RandomSelectionStrategy(), experiment_id)
