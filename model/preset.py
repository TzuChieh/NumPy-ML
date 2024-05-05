from model.network import Network
from model.optimizer import Optimizer


class TrainingPreset:
    def __init__(self):
        self.network: Network = None
        self.optimizer: Optimizer = None
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        self.num_epochs = 1
        self.name = ""

    def train(
        self,
        eval_data=None,
        report_training_performance=False,
        report_eval_performance=False,
        report_training_cost=False,
        report_eval_cost=False):
        """
        Train the network with the specified settings.
        """
        print(self.optimizer.get_info())

        for ei in range(self.num_epochs):
            # Optimize one epoch at a time
            self.optimizer.optimize(self.network, 1, self.training_data)
            
            # Collects a brief report for this epoch
            report = ""

            if report_training_performance:
                num, frac = self.network.performance(self.training_data)
                report += f"; training perf: {num} / {len(self.training_data)} ({frac})"

            if report_eval_performance:
                assert eval_data is not None
                num, frac = self.network.performance(eval_data)
                report += f"; eval perf: {num} / {len(eval_data)} ({frac})"

            if report_training_cost:
                report += f"; training cost: {self.optimizer.total_cost(self.network, self.training_data)}"

            if report_eval_cost:
                assert eval_data is not None
                report += f"; eval cost: {self.optimizer.total_cost(self.network, eval_data)}"

            report += f"; Δt: {self.optimizer.train_time}"

            print(f"epoch {ei + 1} / {self.num_epochs}{report}")

        print(f"total Δt: {self.optimizer.total_train_time}")
