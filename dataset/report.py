import json
from pathlib import Path


class Report:
    def __init__(self):
        self.training_set_size = 0
        self.evaluation_set_size = 0
        self.epoch_to_training_performance = {}
        self.epoch_to_evaluation_performance = {}
        self.epoch_to_training_cost = {}
        self.epoch_to_evaluation_cost = {}
        self.epoch_to_seconds = {}

    def save(self, file_path):
        file_path = Path(file_path).with_suffix('.report')

        # Serialize report in .json
        with open(file_path, 'w') as j:
            j.write(json.dumps(self.__dict__, indent=4))

    def load(self, file_path):
        file_path = Path(file_path).with_suffix('.report')

        # Deserialize report from .json
        with open(file_path, 'r') as f:
            self.__dict__ = json.loads(f.read())

    @staticmethod
    def as_line_series(x_to_y: dict):
        xs = [float(x) for x in x_to_y.keys()]
        ys = [float(y) for y in x_to_y.values()]
        return (xs, ys)
