import dearpygui.dearpygui as dpg
import dearpygui.demo as demo


def trainer_entry_point():
    dpg.create_context()
    dpg.create_viewport(title='Custom Title', width=600, height=600)

    def print_me(sender):
        print(f"Menu Item: {sender}")

    with dpg.viewport_menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Save", callback=print_me)
            dpg.add_menu_item(label="Save As", callback=print_me)

            with dpg.menu(label="Settings"):
                dpg.add_menu_item(label="Setting 1", callback=print_me, check=True)
                dpg.add_menu_item(label="Setting 2", callback=print_me)

        dpg.add_menu_item(label="Help", callback=print_me)

        with dpg.menu(label="Widget Items"):
            dpg.add_checkbox(label="Pick Me", callback=print_me)
            dpg.add_button(label="Press Me", callback=print_me)
            dpg.add_color_picker(label="Color Me", callback=print_me)


    dpg.setup_dearpygui()
    dpg.show_viewport()
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
