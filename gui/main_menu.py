import gui.preset_window
import exp.mnist

import dearpygui.dearpygui as dpg


def _show_mnist_basic_network():
    preset = exp.mnist.load_basic_network_preset()
    gui.preset_window.show(preset)

def show():
    with dpg.viewport_menu_bar():
        with dpg.menu(label="MNIST"):
            dpg.add_menu_item(label="Load Basic Network", callback=_show_mnist_basic_network)
