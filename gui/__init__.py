import gui.main_menu

import dearpygui.dearpygui as dpg
import dearpygui.demo as demo


def trainer_entry_point():
    dpg.create_context()

    # Add a font registry
    with dpg.font_registry():
        # First argument ids the path to the .ttf or .otf file
        default_font = dpg.add_font("./resource/font/Arimo[wght].ttf", 15)

    # Set current font
    dpg.bind_font(default_font)

    dpg.create_viewport(title='Custom Title', width=600, height=600)
    
    gui.main_menu.show()

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.maximize_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


def imgui_demo_entry_point():
    dpg.create_context()
    dpg.create_viewport(title='Custom Title', width=600, height=600)

    demo.show_demo()

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
