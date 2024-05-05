from model.network import Network
from model.optimizer import Optimizer
from dataset.report import Report

from datetime import datetime
from pathlib import Path


class TrainingPreset:
    def __init__(self):
        self.network: Network = None
        self.optimizer: Optimizer = None
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        self.num_epochs = 1
        self.name = ""
        self.report = Report()

    def train(
        self,
        eval_data=None,
        report_training_performance=False,
        report_eval_performance=False,
        report_training_cost=False,
        report_eval_cost=False,
        report_folder_path='./output/'):
        """
        Train the network with the specified settings.
        """
        print(self.optimizer.get_info())

        self.report.training_set_size = len(self.training_data)
        self.report.evaluation_set_size = len(eval_data) if eval_data is not None else 0

        curr_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.report_path = Path(report_folder_path) / curr_time

        for ei in range(self.num_epochs):
            # Optimize one epoch at a time
            self.optimizer.optimize(self.network, 1, self.training_data)
            
            # Collects a brief report for this epoch
            brief = ""

            if report_training_performance:
                num, frac = self.network.performance(self.training_data)
                self.report.epoch_to_training_performance[ei] = frac
                brief += f"; training perf: {num} / {len(self.training_data)} ({frac})"

            if report_eval_performance:
                assert eval_data is not None
                num, frac = self.network.performance(eval_data)
                self.report.epoch_to_evaluation_performance[ei] = frac
                brief += f"; eval perf: {num} / {len(eval_data)} ({frac})"

            if report_training_cost:
                cost = self.optimizer.total_cost(self.network, self.training_data)
                self.report.epoch_to_training_cost[ei] = cost
                brief += f"; training cost: {cost}"

            if report_eval_cost:
                assert eval_data is not None
                cost = self.optimizer.total_cost(self.network, eval_data)
                self.report.epoch_to_evaluation_cost[ei] = cost
                brief += f"; eval cost: {cost}"

            self.report.epoch_to_seconds_spent[ei] = self.optimizer.train_time.total_seconds()
            brief += f"; Δt: {self.optimizer.train_time}"

            if report_folder_path is not None:
                self.report.save(self.report_path)

            print(f"epoch {ei + 1} / {self.num_epochs}{brief}")

        print(f"total Δt: {self.optimizer.total_train_time}")
