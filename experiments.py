import exp.mnist as mnist


def run():
    # preset = mnist.load_basic_network_preset()
    # preset = mnist.load_network_preset()
    preset = mnist.load_deeper_network_preset()
    preset.train(
        eval_data=preset.validation_data,
        report_eval_performance=True,
        report_eval_cost=True)
    preset.network.save("./output/mnist_deeper_40e.model")


if __name__ == '__main__':
    run()
