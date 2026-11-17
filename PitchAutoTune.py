"""Pitch AutoTune Studio - standalone offline GUI prototype.

Install: pip install PyQt6 pyqtgraph numpy
Run:     python PitchAutoTune.py

No SWD/DAPLink implementation is included. Demo data are synthetic, and must
never be treated as validation of controller parameters on physical hardware.
"""

import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSpinBox, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)


@dataclass
class Parameter:
    name: str
    value: float
    low: float
    high: float
    enabled: bool = True
    kind: str = "float"  # float or int
    group: str = "位置环"


DEFAULT_PARAMETERS = [
    Parameter(f"pitch_pos.{n}", v, lo, hi, True, "float", "位置环")
    for n, v, lo, hi in [
        ("Kp", 15, 10, 20), ("Ki", .05, .01, .1),
        ("Kd", .8, .3, 1.3), ("CoefA", 300, 100, 500),
        ("CoefB", 50, 10, 100), ("IntegralLimit", 5000, 2000, 8000),
        ("MaxOut", 100, 80, 120), ("D_LPF", .7, .3, .95),
        ("Output_LPF", .85, .5, .99),
    ]
] + [
    Parameter(f"pitch_speed.{n}", v, lo, hi, True, "float", "速度环")
    for n, v, lo, hi in [
        ("Kp", 8, 5, 12), ("Ki", .02, .005, .05),
        ("Kd", .1, .02, .3), ("CoefA", 200, 80, 350),
        ("CoefB", 30, 5, 75), ("IntegralLimit", 5000, 2000, 8000),
        ("MaxOut", 16000, 10000, 20000), ("D_LPF", .75, .3, .98),
        ("Output_LPF", .9, .5, .99),
    ]
]


def evaluate(t, target, actual, speed, output, band=.3, hold=.3):
    """Compute step metrics with measured timestamps, including irregular sampling."""
    arrays = [np.asarray(x, dtype=float) for x in (t, target, actual, speed, output)]
    t, target, actual, speed, output = arrays
    n = len(t)
    if n < 12 or any(len(x) != n for x in arrays):
        raise ValueError("至少需要 12 个完整采样点")
    if any(not np.all(np.isfinite(x)) for x in arrays):
        raise ValueError("采样数据包含 NaN 或无穷值")
    if not np.all(np.diff(t) > 0):
        raise ValueError("时间戳必须严格递增")
    dt = np.diff(t)
    step_idx = int(np.argmax(np.abs(np.diff(target)))) + 1
    step_amp = float(target[step_idx] - target[max(0, step_idx - 1)])
    if abs(step_amp) < 1e-5:
        raise ValueError("没有检测到目标阶跃；请导入包含目标值变化的数据")
    # This evaluator handles a single step. Multiple steps require separate trials.
    later = np.diff(target[step_idx:])
    if len(later) and np.max(np.abs(later)) > max(abs(step_amp) * .05, .05):
        raise ValueError("检测到多个阶跃：请先拆分 CSV 为单次阶跃实验")
    idx = slice(step_idx, None)
    tr = t[idx] - t[step_idx]
    err = target[idx] - actual[idx]
    post = actual[idx]
    goal = float(np.median(target[step_idx:]))
    signed = np.sign(step_amp)
    initial = float(np.median(actual[:step_idx])) if step_idx else float(actual[0])
    amplitude = max(abs(goal - initial), 1e-8)
    overshoot = max(0., float(np.max(signed * (post - goal)) / amplitude * 100))
    iae = float(np.trapezoid(np.abs(err), tr))
    itae = float(np.trapezoid(tr * np.abs(err), tr))
    # 10%-90% first-crossing rise time (relative to measured initial position).
    projection = signed * (post - initial)
    crossings = []
    for frac in (.1, .9):
        found = np.flatnonzero(projection >= frac * amplitude)
        crossings.append(float(tr[found[0]]) if len(found) else float("nan"))
    rise = crossings[1] - crossings[0] if all(math.isfinite(x) for x in crossings) else float("nan")
    # True settling: the error stays inside the band for the remainder of the trial.
    within = np.abs(err) <= band
    bad = np.flatnonzero(~within)
    settle_idx = int(bad[-1] + 1) if len(bad) else 0
    settling = float(tr[settle_idx]) if settle_idx < len(tr) else float("nan")
    stable = math.isfinite(settling) and (tr[-1] - settling >= hold)
    tail_start = max(step_idx, int(np.searchsorted(t, t[-1] - min(1., (t[-1] - t[step_idx]) / 3))))
    tail = slice(tail_start, None)
    metrics = {
        "rise_ms": rise * 1000,
        "settling_ms": settling * 1000,
        "overshoot_pct": overshoot,
        "iae": iae,
        "itae": itae,
        "steady_error_deg": float(np.mean(np.abs(target[tail] - actual[tail]))),
        "steady_std_deg": float(np.std(actual[tail])),
        "speed_std": float(np.std(speed[tail])),
        "output_rms": float(np.sqrt(np.mean(np.square(output[idx])))),
        "output_std": float(np.std(output[tail])),
        "peak_output": float(np.max(np.abs(output[idx]))),
        "stable": stable,
        "step_time": float(t[step_idx]),
        "settle_absolute": float(t[step_idx] + settling) if math.isfinite(settling) else float("nan"),
        "n": n,
        "median_sample_hz": float(1 / np.median(dt)),
        "max_sample_gap_ms": float(np.max(dt) * 1000),
    }
    return metrics


def score(metrics, baseline, limits):
    """Normalized score, lower is better. Invalid trials are not rankable."""
    if not metrics["stable"] or not math.isfinite(metrics["settling_ms"]):
        return None, "未稳定"
    if metrics["overshoot_pct"] > limits["overshoot"]:
        return None, "超调超限"
    if metrics["peak_output"] > limits["output"]:
        return None, "输出超限"
    if metrics["steady_error_deg"] > limits["steady"]:
        return None, "稳态误差超限"
    keys = [("iae", .40), ("settling_ms", .30),
            ("steady_std_deg", .20), ("output_rms", .10)]
    references = {"iae": .15, "settling_ms": 250., "steady_std_deg": .05,
                  "output_rms": 1000.}
    if baseline:
        references = {k: max(float(baseline[k]), references[k] * .01) for k, _ in keys}
    value = sum(weight * metrics[key] / references[key] for key, weight in keys)
    return round(value, 5), "有效"


def demo_response(start, goal, duration, sample_rate, params, seed):
    """Synthetic illustration, NOT a validated model of an RM gimbal."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / sample_rate)
    t0 = .5
    target = np.where(t < t0, start, goal).astype(float)
    values = {p.name: p.value for p in params}
    kp = values.get("pitch_pos.Kp", 15.)
    kd = values.get("pitch_pos.Kd", .8)
    sp = values.get("pitch_speed.Kp", 8.)
    # Vary toy second-order response to show charts/metrics functionality.
    wn = np.clip(13 * np.sqrt(max(.1, kp / 15) * max(.1, sp / 8)), 4, 24)
    zeta = np.clip(.58 + .15 * (kd / .8 - 1), .2, 1.4)
    actual = np.full(len(t), start, dtype=float)
    speed = np.zeros_like(t)
    dt = 1 / sample_rate
    for i in range(1, len(t)):
        acc = wn * wn * (target[i] - actual[i-1]) - 2 * zeta * wn * speed[i-1]
        speed[i] = speed[i-1] + acc * dt
        actual[i] = actual[i-1] + speed[i] * dt
    actual += rng.normal(0, .015, size=len(t))
    speed_observed = np.gradient(actual, t)
    output = np.clip(180 * (target - actual) - 12 * speed_observed, -15000, 15000)
    return t, target, actual, speed_observed, output


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pitch AutoTune Studio  ·  离线分析 / 自动调参原型")
        self.resize(1500, 900)
        self.parameters = [Parameter(**asdict(p)) for p in DEFAULT_PARAMETERS]
        self.trials = []
        self.current_data = None
        self.current_source = ""
        self.baseline = None
        self.rng = np.random.default_rng()
        self.build_ui()
        self.populate_params()
        self.log("离线模式已就绪。请导入 CSV，或运行演示数据。演示数据不能用于判断实际 PID 优劣。")

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        title = QLabel("PITCH  /  AUTOTUNE STUDIO")
        title.setObjectName("title")
        outer.addWidget(title)
        subtitle = QLabel("任意参数配置  ·  阶跃曲线识别  ·  性能评分  ·  试验归档  ·  局部候选探索    |    通信接口未实现")
        subtitle.setObjectName("muted")
        outer.addWidget(subtitle)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.build_experiment_tab()
        self.build_parameter_tab()
        self.build_history_tab()
        self.build_logs_tab()
        self.statusBar().showMessage("离线测试 · 不连接下位机")
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#111822;color:#e4ebf4;font-size:13px}
            QLabel#title{font-size:24px;font-weight:800;color:#76c7ff;padding:4px}
            QLabel#muted{color:#95a6b9;padding-bottom:8px}
            QGroupBox{border:1px solid #344458;border-radius:7px;margin-top:15px;padding:12px;font-weight:600}
            QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 6px;color:#b9dfff}
            QPushButton{background:#23476b;border:1px solid #37648c;border-radius:5px;padding:7px 12px}
            QPushButton:hover{background:#30618f}
            QPushButton:disabled{color:#76899d;background:#1b2a39}
            QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox,QTextEdit{background:#1b2735;border:1px solid #394d63;border-radius:4px;padding:5px}
            QTableWidget{background:#172331;alternate-background-color:#1b2a3b;gridline-color:#314155;selection-background-color:#275d85}
            QHeaderView::section{background:#26384a;color:#dcebf8;padding:7px;border:0;border-right:1px solid #35475b}
            QTabWidget::pane{border:1px solid #304154}
            QTabBar::tab{background:#1e2b39;padding:10px 18px}
            QTabBar::tab:selected{background:#305d83;color:white}
        """)
        pg.setConfigOptions(antialias=True, background="#172331", foreground="#b5c7dc")

    def build_experiment_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        layout.addLayout(controls)
        settings = QGroupBox("测试设置")
        form = QFormLayout(settings)
        self.start_angle = self.spin(0, -180, 180, 1, " °")
        self.end_angle = self.spin(30, -180, 180, 1, " °")
        self.duration = self.spin(5, 1, 60, 1, " s")
        self.sample_rate = QSpinBox(); self.sample_rate.setRange(10, 5000); self.sample_rate.setValue(200); self.sample_rate.setSuffix(" Hz")
        self.band = self.spin(.3, .01, 20, 2, " °")
        self.max_overshoot = self.spin(20, 0, 500, 1, " %")
        self.max_output = self.spin(16000, 1, 1000000, 0)
        self.max_steady = self.spin(1, .01, 180, 2, " °")
        for label, widget in [
            ("起始角度", self.start_angle), ("目标角度", self.end_angle),
            ("测试时长", self.duration), ("演示采样率", self.sample_rate),
            ("稳定误差带", self.band), ("允许超调", self.max_overshoot),
            ("最大输出绝对值", self.max_output), ("最大稳态误差", self.max_steady),
        ]: form.addRow(label, widget)
        controls.addWidget(settings, 1)
        actions = QGroupBox("数据 / 实验")
        action_layout = QVBoxLayout(actions)
        for title, fn in [
            ("▶ 生成演示曲线", self.run_demo),
            ("导入采样 CSV", self.import_csv),
            ("保存当前实验", self.save_trial),
            ("设为评分基准", self.set_baseline),
            ("生成下一组候选参数", self.propose_candidate),
            ("导出当前 CSV", self.export_csv),
        ]:
            b = QPushButton(title); b.clicked.connect(fn); action_layout.addWidget(b)
        action_layout.addStretch()
        hint = QLabel("候选生成器仅做局部探索；\n未实现真机通信和安全保护。\n评分较低不等于真机更安全。")
        hint.setWordWrap(True)
        action_layout.addWidget(hint)
        controls.addWidget(actions, 1)
        self.metrics_box = QGroupBox("性能指标")
        metric_layout = QGridLayout(self.metrics_box)
        self.metric_labels = {}
        names = ["评分", "有效性", "上升时间", "调节时间", "超调量", "IAE", "ITAE", "稳态误差", "稳态抖动", "输出峰值", "稳态输出波动", "采样率"]
        for index, name in enumerate(names):
            k = QLabel(name); v = QLabel("—"); v.setStyleSheet("font-size:17px;color:#7dd7ff;font-weight:700")
            row, col = divmod(index, 3)
            metric_layout.addWidget(k, row, col * 2)
            metric_layout.addWidget(v, row, col * 2 + 1)
            self.metric_labels[name] = v
        controls.addWidget(self.metrics_box, 2)

        plots = QSplitter(Qt.Orientation.Vertical)
        self.angle_plot = pg.PlotWidget(title="Pitch 目标 / 反馈  (°)")
        self.speed_plot = pg.PlotWidget(title="角速度  (°/s)")
        self.output_plot = pg.PlotWidget(title="控制输出")
        for p in (self.angle_plot, self.speed_plot, self.output_plot):
            p.showGrid(x=True, y=True, alpha=.2)
            p.setLabel("bottom", "时间", units="s")
            p.getPlotItem().setMenuEnabled(False)
            plots.addWidget(p)
        self.angle_plot.addLegend()
        self.angle_plot.setXLink(self.speed_plot)
        self.output_plot.setXLink(self.speed_plot)
        layout.addWidget(plots, 1)
        self.tabs.addTab(page, "实验与曲线")

    def build_parameter_tab(self):
        page = QWidget(); layout = QVBoxLayout(page)
        info = QLabel("双击单元格编辑。任意添加参数；勾选“优化”才参与候选生成。参数范围由用户确认；本软件不推断安全范围。")
        info.setWordWrap(True); layout.addWidget(info)
        self.param_table = QTableWidget(0, 7)
        self.param_table.setHorizontalHeaderLabels(["优化", "分组", "变量名 / ELF 符号", "当前值", "下限", "上限", "类型"])
        self.param_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.param_table.setAlternatingRowColors(True)
        layout.addWidget(self.param_table)
        buttons = QHBoxLayout()
        for label, func in [
            ("添加参数", self.add_parameter), ("删除选中参数", self.remove_parameter),
            ("载入 JSON 配置", self.load_config), ("保存 JSON 配置", self.save_config),
            ("应用表格修改", self.commit_params),
        ]:
            b = QPushButton(label); b.clicked.connect(func); buttons.addWidget(b)
        layout.addLayout(buttons)
        self.tabs.addTab(page, "参数空间")

    def build_history_tab(self):
        page = QWidget(); layout = QVBoxLayout(page)
        self.history_table = QTableWidget(0, 10)
        self.history_table.setHorizontalHeaderLabels(["编号", "来源", "评分", "状态", "超调 %", "调节 ms", "IAE", "稳态 σ", "输出峰值", "时间"])
        self.history_table.setAlternatingRowColors(True)
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.history_table.cellDoubleClicked.connect(self.show_trial)
        layout.addWidget(self.history_table)
        buttons = QHBoxLayout()
        for label, fn in [
            ("导出实验 JSON", self.export_history),
            ("载入实验 JSON", self.import_history),
            ("对比选中实验曲线", self.compare_history),
        ]:
            btn = QPushButton(label); btn.clicked.connect(fn); buttons.addWidget(btn)
        layout.addLayout(buttons)
        layout.addWidget(QLabel("双击记录查看该次曲线；Ctrl/Shift 多选两行可叠加比较。全部实验包含当时的参数快照。"))
        self.tabs.addTab(page, "实验历史")

    def build_logs_tab(self):
        page = QWidget(); layout = QVBoxLayout(page)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        self.tabs.addTab(page, "日志")

    @staticmethod
    def spin(value, low, high, decimals, suffix=""):
        s = QDoubleSpinBox(); s.setRange(low, high); s.setDecimals(decimals)
        s.setValue(value); s.setSuffix(suffix); return s

    def log(self, msg):
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def populate_params(self):
        self.param_table.setRowCount(len(self.parameters))
        for row, p in enumerate(self.parameters):
            checkbox = QTableWidgetItem(); checkbox.setFlags(checkbox.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checkbox.setCheckState(Qt.CheckState.Checked if p.enabled else Qt.CheckState.Unchecked)
            self.param_table.setItem(row, 0, checkbox)
            for col, value in enumerate((p.group, p.name, p.value, p.low, p.high, p.kind), 1):
                self.param_table.setItem(row, col, QTableWidgetItem(str(value)))

    def commit_params(self):
        params = []; seen = set()
        try:
            for row in range(self.param_table.rowCount()):
                cells = [self.param_table.item(row, col) for col in range(7)]
                vals = [c.text().strip() if c else "" for c in cells]
                kind = vals[6].lower()
                if kind not in ("float", "int"):
                    raise ValueError(f"第 {row+1} 行：类型只能是 float/int")
                p = Parameter(vals[2], float(vals[3]), float(vals[4]), float(vals[5]),
                              cells[0].checkState() == Qt.CheckState.Checked, kind, vals[1])
                if not p.name or p.name in seen: raise ValueError(f"第 {row+1} 行：变量名为空或重复")
                if not all(math.isfinite(v) for v in (p.value, p.low, p.high)):
                    raise ValueError(f"{p.name}: 存在无效数值")
                if not p.low <= p.value <= p.high or p.low >= p.high:
                    raise ValueError(f"{p.name}: 需要下限 ≤ 当前值 ≤ 上限，且上限 > 下限")
                if kind == "int" and any(not x.is_integer() for x in (p.value, p.low, p.high)):
                    raise ValueError(f"{p.name}: 整数参数必须使用整数值")
                seen.add(p.name); params.append(p)
        except (ValueError, AttributeError) as exc:
            QMessageBox.warning(self, "参数校验失败", str(exc)); return False
        self.parameters = params
        self.log(f"已确认 {len(params)} 个参数，其中 {sum(p.enabled for p in params)} 个参与探索")
        return True

    def add_parameter(self):
        row = self.param_table.rowCount(); self.param_table.insertRow(row)
        values = ["", "自定义", f"custom.parameter_{row+1}", "0.5", "0", "1", "float"]
        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            if col == 0: item.setCheckState(Qt.CheckState.Checked)
            self.param_table.setItem(row, col, item)
        self.param_table.scrollToBottom()

    def remove_parameter(self):
        for row in sorted({i.row() for i in self.param_table.selectedIndexes()}, reverse=True):
            self.param_table.removeRow(row)

    def save_config(self):
        if not self.commit_params(): return
        path, _ = QFileDialog.getSaveFileName(self, "保存参数配置", "pitch_parameters.json", "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps([asdict(p) for p in self.parameters], ensure_ascii=False, indent=2), encoding="utf-8")
            self.log(f"保存参数配置：{path}")

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "载入参数配置", "", "JSON (*.json)")
        if not path: return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.parameters = [Parameter(**p) for p in data]
            self.populate_params()
            if not self.commit_params(): raise ValueError("配置校验不通过")
            self.log(f"载入配置：{path}")
        except (ValueError, KeyError, TypeError) as e: QMessageBox.warning(self, "载入失败", str(e))

    def run_demo(self):
        if not self.commit_params(): return
        if abs(self.start_angle.value()-self.end_angle.value()) < .01:
            QMessageBox.warning(self, "无效轨迹", "起始角度与目标角度需要不同"); return
        data = demo_response(self.start_angle.value(), self.end_angle.value(),
                             self.duration.value(), self.sample_rate.value(),
                             self.parameters, len(self.trials) + int(time.time()) % 10000)
        self.accept_data(data, "模拟数据（非真机）")

    def import_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入采样 CSV", "", "CSV (*.csv)")
        if not path: return
        try:
            with open(path, encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                required = ("time", "target", "actual", "speed", "output")
                if not reader.fieldnames or any(c not in reader.fieldnames for c in required):
                    raise ValueError("CSV 表头必须有 time,target,actual,speed,output；time 单位为秒")
                rows = [[float(row[key]) for key in required] for row in reader]
            arr = np.asarray(rows, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 5: raise ValueError("没有有效的五列采样数据")
            self.accept_data(tuple(arr[:, i] for i in range(5)), Path(path).name)
        except (ValueError, OSError, KeyError) as e: QMessageBox.warning(self, "CSV 读取失败", str(e))

    def limits(self):
        return {"overshoot": self.max_overshoot.value(), "output": self.max_output.value(),
                "steady": self.max_steady.value()}

    def accept_data(self, data, source):
        try:
            metrics = evaluate(*data, band=self.band.value())
            result, reason = score(metrics, self.baseline, self.limits())
        except ValueError as e:
            QMessageBox.warning(self, "分析失败", str(e)); return
        self.current_data = tuple(np.array(x, dtype=float) for x in data)
        self.current_source = source
        self.current_metrics = metrics
        self.current_score = result
        self.current_reason = reason
        self.draw_curves(self.current_data)
        labels = {
            "评分": f"{result:.4f}" if result is not None else "—",
            "有效性": reason, "上升时间": self.fmt(metrics["rise_ms"], " ms"),
            "调节时间": self.fmt(metrics["settling_ms"], " ms"),
            "超调量": self.fmt(metrics["overshoot_pct"], " %"),
            "IAE": f'{metrics["iae"]:.5f}', "ITAE": f'{metrics["itae"]:.5f}',
            "稳态误差": self.fmt(metrics["steady_error_deg"], " °"),
            "稳态抖动": self.fmt(metrics["steady_std_deg"], " °"),
            "输出峰值": self.fmt(metrics["peak_output"], ""),
            "稳态输出波动": self.fmt(metrics["output_std"], ""),
            "采样率": self.fmt(metrics["median_sample_hz"], " Hz"),
        }
        for key, value in labels.items(): self.metric_labels[key].setText(value)
        self.log(f"分析 {source}：{reason}；得分={result}；最大采样间隔 {metrics['max_sample_gap_ms']:.1f} ms")

    @staticmethod
    def fmt(value, suffix):
        return f"{value:.2f}{suffix}" if math.isfinite(value) else "未达到"

    def draw_curves(self, data, overlay=None):
        for plot in (self.angle_plot, self.speed_plot, self.output_plot): plot.clear()
        t, target, actual, speed, output = data
        self.angle_plot.plot(t, target, pen=pg.mkPen("#ffcb6b", width=2, style=Qt.PenStyle.DashLine), name="目标")
        self.angle_plot.plot(t, actual, pen=pg.mkPen("#67d5fb", width=2), name="实际")
        self.speed_plot.plot(t, speed, pen=pg.mkPen("#96e6a1", width=2))
        self.output_plot.plot(t, output, pen=pg.mkPen("#d5a6ff", width=2))
        if overlay is not None:
            a, b, c, d, e = overlay
            self.angle_plot.plot(a, c, pen=pg.mkPen("#ff768d", width=2))
            self.speed_plot.plot(a, d, pen=pg.mkPen("#ff768d", width=2))
            self.output_plot.plot(a, e, pen=pg.mkPen("#ff768d", width=2))
        if self.current_data is not None:
            try:
                m = evaluate(*data, band=self.band.value())
                for p in (self.angle_plot, self.speed_plot, self.output_plot):
                    p.addItem(pg.InfiniteLine(pos=m["step_time"], angle=90,
                                              pen=pg.mkPen("#ffcb6b", style=Qt.PenStyle.DotLine)))
                if math.isfinite(m["settle_absolute"]):
                    self.angle_plot.addItem(pg.InfiniteLine(pos=m["settle_absolute"], angle=90,
                                             pen=pg.mkPen("#62d6a0", style=Qt.PenStyle.DotLine)))
            except ValueError: pass

    def save_trial(self):
        if self.current_data is None:
            QMessageBox.information(self, "暂无数据", "请先导入或生成曲线"); return
        if not self.commit_params(): return
        trial = {
            "id": len(self.trials) + 1, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": self.current_source, "parameters": [asdict(p) for p in self.parameters],
            "metrics": dict(self.current_metrics), "score": self.current_score,
            "status": self.current_reason,
            "data": [arr.tolist() for arr in self.current_data],
        }
        self.trials.append(trial)
        self.update_history()
        self.log(f"保存实验 #{trial['id']}，含 {len(self.parameters)} 个参数快照")

    def set_baseline(self):
        if self.current_data is None: return
        if not self.current_metrics["stable"]:
            QMessageBox.warning(self, "基准无效", "当前曲线尚未稳定，不能作为评分基准"); return
        self.baseline = dict(self.current_metrics)
        self.accept_data(self.current_data, self.current_source)
        self.log("已设置基准指标；后续得分以该指标归一化（越小越好）")

    def propose_candidate(self):
        if not self.commit_params(): return
        if not self.trials:
            QMessageBox.information(self, "缺少实验", "请先保存至少一组真实或演示实验作为候选生成的起点"); return
        # Conservative local proposal, not BO. Using demo trials does not provide real evidence.
        valid = [x for x in self.trials if x["score"] is not None]
        best = min(valid, key=lambda x: x["score"]) if valid else self.trials[-1]
        best_by_name = {p["name"]: p["value"] for p in best["parameters"]}
        changed = 0
        for p in self.parameters:
            if not p.enabled: continue
            anchor = best_by_name.get(p.name, p.value)
            span = p.high - p.low
            suggestion = float(np.clip(anchor + self.rng.normal(0, .05 * span), p.low, p.high))
            if p.kind == "int": suggestion = float(round(suggestion))
            if suggestion != p.value: changed += 1
            p.value = suggestion
        self.populate_params()
        self.tabs.setCurrentIndex(1)
        self.log(f"已围绕实验 #{best['id']} 生成 {changed} 个参数候选变化；请人工检查约束后再用于真机")
        QMessageBox.information(self, "局部候选已生成", "候选已填写到参数表。\n\n这是随机局部探索原型，不是贝叶斯优化，\n不会自动下发到下位机，也不包含跨参数安全约束。")

    def update_history(self):
        self.history_table.setRowCount(len(self.trials))
        for row, tr in enumerate(self.trials):
            m = tr["metrics"]
            vals = [tr["id"], tr["source"], tr["score"] if tr["score"] is not None else "—",
                    tr["status"], round(m["overshoot_pct"], 2),
                    round(m["settling_ms"], 2) if math.isfinite(m["settling_ms"]) else "—",
                    round(m["iae"], 4), round(m["steady_std_deg"], 4),
                    round(m["peak_output"], 1), tr["time"]]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(str(val)); item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.history_table.setItem(row, col, item)

    def show_trial(self, row, _col):
        if row >= len(self.trials): return
        tr = self.trials[row]
        self.accept_data(tuple(np.array(x) for x in tr["data"]), tr["source"])
        self.tabs.setCurrentIndex(0)

    def compare_history(self):
        rows = sorted({i.row() for i in self.history_table.selectedIndexes()})
        if len(rows) != 2:
            QMessageBox.information(self, "选择两组", "请在历史表中按 Ctrl 选择两行"); return
        a, b = (tuple(np.array(x) for x in self.trials[r]["data"]) for r in rows)
        self.current_data = a
        self.draw_curves(a, b)
        self.tabs.setCurrentIndex(0)
        self.log(f"正在叠加对比实验 #{rows[0]+1}（蓝色）与 #{rows[1]+1}（粉色）")

    def export_csv(self):
        if self.current_data is None: return
        path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", "pitch_samples.csv", "CSV (*.csv)")
        if not path: return
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f); w.writerow(("time", "target", "actual", "speed", "output"))
            w.writerows(zip(*self.current_data))
        self.log(f"导出采样：{path}")

    def export_history(self):
        if not self.trials: return
        path, _ = QFileDialog.getSaveFileName(self, "导出实验历史", "pitch_trials.json", "JSON (*.json)")
        if path:
            payload = {"version": 1, "baseline": self.baseline, "trials": self.trials}
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.log(f"导出 {len(self.trials)} 组实验：{path}")

    def import_history(self):
        path, _ = QFileDialog.getOpenFileName(self, "载入实验历史", "", "JSON (*.json)")
        if not path: return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if payload.get("version") != 1: raise ValueError("不支持的实验历史版本")
            trials = payload["trials"]
            if not isinstance(trials, list): raise ValueError("无效的实验记录")
            for tr in trials:
                if not all(k in tr for k in ("metrics", "parameters", "data", "source", "time", "score", "status")):
                    raise ValueError("实验记录缺少必要字段")
                evaluate(*(np.asarray(x) for x in tr["data"]))
            self.trials = trials
            self.baseline = payload.get("baseline")
            self.update_history()
            self.log(f"载入实验历史：{path}")
        except (ValueError, KeyError, TypeError, OSError) as e:
            QMessageBox.warning(self, "导入失败", str(e))


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
