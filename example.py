import exp.mnist as mnist

from pathlib import Path


def run():
    preset = mnist.load_basic_network_preset()

    # Train the network using default settings
    preset.train(
        eval_set=preset.test_set,
        report_eval_performance=True,
        report_eval_cost=True)
    
    # Save the trained network
    preset.network.save(Path("./output/") / preset.name)

if __name__ == '__main__':
    run()
