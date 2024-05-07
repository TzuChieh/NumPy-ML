import gui.training_report

import dearpygui.dearpygui as dpg


def add():
    with dpg.viewport_menu_bar():
        with dpg.menu(label="View Report"):
            dpg.add_menu_item(label="Training Report", callback=gui.training_report.show)

    