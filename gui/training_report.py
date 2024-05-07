from dataset.report import Report

import dearpygui.dearpygui as dpg

from pathlib import Path


def add():
    with dpg.value_registry():
        dpg.add_string_value(tag='report_folder', default_value=Path("./output/").absolute())
        dpg.add_string_value(tag='report_path', default_value="")
        dpg.add_bool_value(tag='auto_fit', default_value=True)

    def _on_report_folder_confirmed(sender, app_data):
        folder_path = app_data['file_path_name']
        print(f"selecting reports from {folder_path}")
        dpg.set_value('report_folder', folder_path)
        _update_report_list()

    dpg.add_file_dialog(
        tag='report_folder_dialog',
        directory_selector=True,
        show=False,
        callback=_on_report_folder_confirmed,
        width=700,
        height=400)

    with dpg.window(label="Training Report", width=1440, height=900, no_scrollbar=False, horizontal_scrollbar=True, show=False, tag='training_report'):
        with dpg.group(horizontal=True):
            with dpg.child_window(tag='reports', width=300):
                _show_reports_window_content()
            with dpg.child_window(tag='viewer', autosize_x=True):
                _show_viewer_window_content()

def show():
    dpg.show_item('training_report')

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
    dpg.add_listbox(tag='report_list', width=-1, num_items=20, callback=_on_report_file_selected)
    _update_report_list()

def _show_viewer_window_content():
    def _on_load_report(sender, app_data, user_data: Report):
        report_path = Path(dpg.get_value('report_path'))
        user_data.load(report_path)

        dpg.set_value(
            'epoch_to_training_performance', Report.as_line_series(user_data.epoch_to_training_performance))
        dpg.set_value(
            'epoch_to_evaluation_performance', Report.as_line_series(user_data.epoch_to_evaluation_performance))
        dpg.set_value(
            'epoch_to_training_cost', Report.as_line_series(user_data.epoch_to_training_cost))
        dpg.set_value(
            'epoch_to_evaluation_cost', Report.as_line_series(user_data.epoch_to_evaluation_cost))
        dpg.set_value(
            'epoch_to_seconds', Report.as_line_series(user_data.epoch_to_seconds))
        
        auto_fit = dpg.get_value('auto_fit')
        if auto_fit:
            dpg.fit_axis_data('epoch_performance_x')
            dpg.fit_axis_data('epoch_performance_y')
            dpg.fit_axis_data('epoch_seconds_x')
            dpg.fit_axis_data('epoch_seconds_y')
            dpg.fit_axis_data('epoch_cost_x')
            dpg.fit_axis_data('epoch_cost_y')

    dpg.add_input_text(label="Selected Report", source='report_path')
    with dpg.group(horizontal=True):
        report = Report()
        dpg.add_button(label="Load", user_data=report, callback=_on_load_report)
        dpg.add_checkbox(label="Auto Fit", source='auto_fit')
    
    with dpg.plot(label="##epoch-performance", height=350, width=-1):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="Epoch", tag='epoch_performance_x')
        with dpg.plot_axis(dpg.mvYAxis, label="Performance", tag='epoch_performance_y'):
            dpg.add_line_series([], [], label="Training", tag='epoch_to_training_performance')
            dpg.add_line_series([], [], label="Evaluation", tag='epoch_to_evaluation_performance')

    with dpg.plot(label="##epoch-cost", height=350, width=-1):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="Epoch", tag='epoch_cost_x')
        with dpg.plot_axis(dpg.mvYAxis, label="Cost", tag='epoch_cost_y'):
            dpg.add_line_series([], [], label="Training", tag='epoch_to_training_cost')
            dpg.add_line_series([], [], label="Evaluation", tag='epoch_to_evaluation_cost')

    with dpg.plot(label="##epoch-seconds", height=350, width=-1):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="Epoch", tag='epoch_seconds_x')
        with dpg.plot_axis(dpg.mvYAxis, label="Seconds", tag='epoch_seconds_y'):
            dpg.add_line_series([], [], tag='epoch_to_seconds')

def _update_report_list():
    report_folder = Path(dpg.get_value('report_folder'))
    files = [p.relative_to(report_folder) for p in report_folder.iterdir() if p.suffix == ".report"]
    dpg.configure_item('report_list', items=files)
