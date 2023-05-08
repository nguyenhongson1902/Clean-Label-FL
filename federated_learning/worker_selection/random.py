from .selection_strategy import SelectionStrategy
import random

class RandomSelectionStrategy(SelectionStrategy):
    """
    Randomly selects workers out of the list of all workers
    """

    def select_round_workers(self, workers, poisoned_workers, kwargs):
        return random.sample(workers, kwargs["NUM_WORKERS_PER_ROUND"])
    
    def select_round_workers_and_malicious_client(self, workers, poisoned_workers, kwargs):
        random_workers = random.sample(workers, kwargs["NUM_WORKERS_PER_ROUND"] - 1)
        random_workers.append(random.sample(poisoned_workers, 1)[0])
        return random_workers
