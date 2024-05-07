import exp.mnist as mnist


def run():
    # preset = mnist.load_basic_network_preset()
    # preset = mnist.load_network_preset()
    preset = mnist.load_deeper_network_preset()
    preset.network.save("./output/test2.model")
    preset.network.load("./output/test2.model")
    # preset.train(
    #     eval_data=preset.validation_data,
    #     report_eval_performance=True,
    #     report_eval_cost=True)


if __name__ == '__main__':
    run()
