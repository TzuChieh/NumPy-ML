import gui.training_report_viewer
import gui.model_viewer

import dearpygui.dearpygui as dpg


def add():
    with dpg.viewport_menu_bar():
        with dpg.menu(label="View Report"):
            dpg.add_menu_item(label="Training Report Viewer", callback=gui.training_report_viewer.show)
        with dpg.menu(label="View Model"):
            dpg.add_menu_item(label="Model Viewer", callback=gui.model_viewer.show)

    