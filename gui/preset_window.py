from model.preset import TrainingPreset

import dearpygui.dearpygui as dpg


def show(preset: TrainingPreset):
    def _on_window_close():
        pass

    with dpg.window(label=preset.name, width=800, height=800, on_close=_on_window_close):
        pass
