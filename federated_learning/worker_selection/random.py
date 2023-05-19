from .selection_strategy import SelectionStrategy
import random

class RandomSelectionStrategy(SelectionStrategy):
    """
    Randomly selects workers out of the list of all workers
    """

    def select_round_workers(self, workers, poisoned_workers, kwargs):
        return random.sample(workers, kwargs["NUM_WORKERS_PER_ROUND"])
    
    def select_round_workers_and_malicious_client(self, workers, poisoned_workers, kwargs):
        new_workers = [worker for worker in workers if worker not in poisoned_workers]
        assert len(new_workers) >= kwargs["NUM_WORKERS_PER_ROUND"] - len(poisoned_workers)

        random_workers = random.sample(new_workers, kwargs["NUM_WORKERS_PER_ROUND"] - len(poisoned_workers))
        random_workers.extend(poisoned_workers)
        return random_workers
    
    def select_2_poisoned_clients(self, workers, poisoned_workers, kwargs):
        return random.sample(poisoned_workers, 2)
    
    def select_fixed_frequency():
        pass

    def select_fixed_pool():
        pass
