import exp.mnist as mnist
import exp.fashion_mnist as fashion_mnist

from pathlib import Path


def run():
    preset = mnist.load_basic_network_preset()
    # preset = mnist.load_network_preset()
    # preset = mnist.load_deeper_network_preset()

    # preset = fashion_mnist.load_basic_network_preset()
    # preset = fashion_mnist.load_network_preset()

    # Train the network using default settings
    preset.train(
        eval_set=preset.validation_set,
        report_eval_performance=True,
        report_eval_cost=True)
    
    # Save the trained network
    preset.network.save(Path("./output/") / preset.name)


if __name__ == '__main__':
    run()
