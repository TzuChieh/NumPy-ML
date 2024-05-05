import exp.mnist as mnist


def run():
    # preset = mnist.load_basic_network_preset()
    preset = mnist.load_network_preset()
    preset.train(
        eval_data=preset.validation_data,
        report_eval_performance=True,
        report_eval_cost=True)


if __name__ == '__main__':
    run()
