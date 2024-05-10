from model.network import Network
from model.optimizer import Optimizer
from dataset import Dataset
from dataset.report import Report

from datetime import datetime
from pathlib import Path
from datetime import timedelta
from timeit import default_timer as timer


class TrainingPreset:
    def __init__(self):
        self.network: Network = None
        self.optimizer: Optimizer = None
        self.training_set: Dataset=None
        self.validation_set: Dataset=None
        self.test_set: Dataset=None
        self.num_epochs = 1
        self.name = ""
        self.report = Report()

    def train(
        self,
        eval_set: Dataset=None,
        report_training_performance=False,
        report_eval_performance=False,
        report_training_cost=False,
        report_eval_cost=False,
        report_folder_path='./output/'):
        """
        Train the network with the specified settings.
        """
        print(self.optimizer.get_info())

        self.report.training_set_size = len(self.training_set)
        self.report.evaluation_set_size = len(eval_set) if eval_set is not None else 0

        curr_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.report_path = Path(report_folder_path) / curr_time

        for ei in range(self.num_epochs):
            print(f"Epoch {ei + 1} / {self.num_epochs}:")

            # Optimize one epoch at a time
            self.optimizer.optimize(self.network, 1, self.training_set, print_progress=True)
            
            # Collects a brief report for this epoch
            brief = ""
            report_start_time = timer()

            if report_training_performance:
                num, frac = self.optimizer.performance(self.training_set, self.network)
                self.report.epoch_to_training_performance[ei] = frac
                brief += f" | training perf: {num} / {len(self.training_set)} ({frac:.4f})"

            if report_eval_performance:
                assert eval_set is not None
                num, frac = self.optimizer.performance(eval_set, self.network)
                self.report.epoch_to_evaluation_performance[ei] = frac
                brief += f" | eval perf: {num} / {len(eval_set)} ({frac:.4f})"

            if report_training_cost:
                cost = self.optimizer.total_cost(self.training_set, self.network)
                self.report.epoch_to_training_cost[ei] = cost
                brief += f" | training cost: {cost:.4f}"

            if report_eval_cost:
                assert eval_set is not None
                cost = self.optimizer.total_cost(eval_set, self.network)
                self.report.epoch_to_evaluation_cost[ei] = cost
                brief += f" | eval cost: {cost:.4f}"

            self.report.epoch_to_seconds[ei] = self.optimizer.total_train_time.total_seconds()
            brief += f" | Δt_epoch: {self.optimizer.train_time}"
            brief += f" | Δt_report: {timedelta(seconds=(timer() - report_start_time))}" if brief else ""

            if report_folder_path is not None:
                self.report.save(self.report_path)

            print(f"{brief}")

        print(f"total Δt: {self.optimizer.total_train_time}")
