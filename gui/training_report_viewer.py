from dataset.report import Report

import dearpygui.dearpygui as dpg

from pathlib import Path


class Context:
    def __init__(self):
        self.report_folder_dialog = None
        self.main_window = None
        self.reports_window = None
        self.viewer_window = None
        self.report_folder = None
        self.report_path = None
        self.report_list = None
        self.auto_fit = None
        self.epoch_performance_x = None
        self.epoch_performance_y = None
        self.epoch_to_training_performance = None
        self.epoch_to_evaluation_performance = None
        self.epoch_cost_x = None
        self.epoch_cost_y = None
        self.epoch_to_training_cost = None
        self.epoch_to_evaluation_cost = None
        self.epoch_seconds_x = None
        self.epoch_seconds_y = None
        self.epoch_to_seconds = None



ctx = Context()


def add():
    global ctx

    def _on_report_folder_confirmed(sender, app_data):
        folder_path = app_data['file_path_name']
        print(f"selecting reports from {folder_path}")
        dpg.set_value(ctx.report_folder, folder_path)
        _update_report_list(ctx)

    ctx.report_folder_dialog = dpg.add_file_dialog(
        directory_selector=True,
        show=False,
        callback=_on_report_folder_confirmed,
        width=700,
        height=400)

    with dpg.window(label="Training Report Viewer", width=1440, height=900, no_scrollbar=False, horizontal_scrollbar=True, show=False) as window:
        ctx.main_window = window
        with dpg.group(horizontal=True):
            with dpg.child_window(width=300) as window:
                ctx.reports_window = window
                _show_reports_window_content(ctx)
            with dpg.child_window(autosize_x=True) as window:
                ctx.viewer_window = window
                _show_viewer_window_content(ctx)

def show():
    global ctx

    dpg.show_item(ctx.main_window)

def _show_reports_window_content(ctx: Context):
    def _on_select_report_folder():
        dpg.show_item(ctx.report_folder_dialog)

    def _on_report_file_selected(sender, app_data):
        report_path = Path(dpg.get_value(ctx.report_folder)) / app_data
        dpg.set_value(ctx.report_path, report_path)

    with dpg.group(horizontal=True):
        dpg.add_text("Report Folder")
        dpg.add_button(label="Select", callback=_on_select_report_folder)

    ctx.report_folder = dpg.add_input_text(label="##report-folder", default_value=Path("./output/").absolute(), width=-1, enabled=False)

    dpg.add_separator()

    dpg.add_text("Reports:")
    ctx.report_list = dpg.add_listbox(width=-1, num_items=20, callback=_on_report_file_selected)
    _update_report_list(ctx)

def _show_viewer_window_content(ctx: Context):
    def _on_load_report(sender, app_data, user_data: Report):
        report_path = Path(dpg.get_value(ctx.report_path))
        user_data.load(report_path)

        dpg.set_value(
            ctx.epoch_to_training_performance, Report.as_line_series(user_data.epoch_to_training_performance))
        dpg.set_value(
            ctx.epoch_to_evaluation_performance, Report.as_line_series(user_data.epoch_to_evaluation_performance))
        dpg.set_value(
            ctx.epoch_to_training_cost, Report.as_line_series(user_data.epoch_to_training_cost))
        dpg.set_value(
            ctx.epoch_to_evaluation_cost, Report.as_line_series(user_data.epoch_to_evaluation_cost))
        dpg.set_value(
            ctx.epoch_to_seconds, Report.as_line_series(user_data.epoch_to_seconds))
        
        auto_fit = dpg.get_value(ctx.auto_fit)
        if auto_fit:
            dpg.fit_axis_data(ctx.epoch_performance_x)
            dpg.fit_axis_data(ctx.epoch_performance_y)
            dpg.fit_axis_data(ctx.epoch_seconds_x)
            dpg.fit_axis_data(ctx.epoch_seconds_y)
            dpg.fit_axis_data(ctx.epoch_cost_x)
            dpg.fit_axis_data(ctx.epoch_cost_y)

    ctx.report_path = dpg.add_input_text(label="Selected Report")
    with dpg.group(horizontal=True):
        report = Report()
        dpg.add_button(label="Load", user_data=report, callback=_on_load_report)
        ctx.auto_fit = dpg.add_checkbox(label="Auto Fit", default_value=True)
    
    with dpg.plot(label="##epoch-performance", height=350, width=-1):
        dpg.add_plot_legend()
        ctx.epoch_performance_x = dpg.add_plot_axis(dpg.mvXAxis, label="Epoch")
        with dpg.plot_axis(dpg.mvYAxis, label="Performance") as y_axis:
            ctx.epoch_performance_y = y_axis
            ctx.epoch_to_training_performance = dpg.add_line_series([], [], label="Training")
            ctx.epoch_to_evaluation_performance = dpg.add_line_series([], [], label="Evaluation")
    
    with dpg.plot(label="##epoch-cost", height=350, width=-1):
        dpg.add_plot_legend()
        ctx.epoch_cost_x = dpg.add_plot_axis(dpg.mvXAxis, label="Epoch")
        with dpg.plot_axis(dpg.mvYAxis, label="Cost") as y_axis:
            ctx.epoch_cost_y = y_axis
            ctx.epoch_to_training_cost = dpg.add_line_series([], [], label="Training")
            ctx.epoch_to_evaluation_cost = dpg.add_line_series([], [], label="Evaluation")

    with dpg.plot(label="##epoch-seconds", height=350, width=-1):
        dpg.add_plot_legend()
        ctx.epoch_seconds_x = dpg.add_plot_axis(dpg.mvXAxis, label="Epoch")
        with dpg.plot_axis(dpg.mvYAxis, label="Seconds") as y_axis:
            ctx.epoch_seconds_y = y_axis
            ctx.epoch_to_seconds = dpg.add_line_series([], [])

def _update_report_list(ctx: Context):
    report_folder = Path(dpg.get_value(ctx.report_folder))
    files = [p.relative_to(report_folder) for p in report_folder.iterdir() if p.suffix == ".report"]
    dpg.configure_item(ctx.report_list, items=files)
