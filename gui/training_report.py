import dearpygui.dearpygui as dpg

from pathlib import Path


def show():
    with dpg.value_registry():
        dpg.add_string_value(tag='report_folder', default_value=Path("./output/").absolute())
        dpg.add_string_value(tag='report_path', default_value="")

    def _on_report_folder_confirmed(sender, app_data):
        folder_path = Path(app_data['file_path_name'])
        print(f"selecting reports from {folder_path}")
        dpg.set_value('report_folder', folder_path)

        files = [p.relative_to(folder_path) for p in folder_path.iterdir() if p.suffix == ".report"]
        dpg.configure_item('report_list', items=files)

    dpg.add_file_dialog(
        tag='report_folder_dialog',
        directory_selector=True,
        show=False,
        callback=_on_report_folder_confirmed,
        width=700,
        height=400)

    with dpg.window(label="Training Report", width=1280, height=720, no_scrollbar=False, horizontal_scrollbar=True):
        with dpg.group(horizontal=True):
            with dpg.child_window(tag='reports', width=300):
                _show_reports_window_content()
            with dpg.child_window(tag='viewer', autosize_x=True):
                _show_viewer_window_content()

def _show_reports_window_content():
    def _on_select_report_folder():
        dpg.show_item('report_folder_dialog')

    def _on_report_file_selected(sender, app_data):
        report_path = Path(dpg.get_value('report_folder')) / app_data
        dpg.set_value('report_path', report_path)

    with dpg.group(horizontal=True):
        dpg.add_text("Report Folder")
        dpg.add_button(label="Select", callback=_on_select_report_folder)

    dpg.add_input_text(label="##report-folder", source='report_folder', width=-1, enabled=False)

    dpg.add_separator()

    dpg.add_text("Reports:")
    dpg.add_listbox(tag='report_list', callback=_on_report_file_selected)

def _show_viewer_window_content():
    dpg.add_input_text(label="Selected Report", source='report_path')
