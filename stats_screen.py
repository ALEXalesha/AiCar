"""Экран статистики (2.0.0): что накопилось в stats.json, по образцу Нейро-змейки.

Сверху карточки с крупными числами, ниже таблицы по уровням и по генераторам трассы,
сравнение с замером автора, машинки, графики. Всё считается из `stats.load_totals()`:
общие счётчики были и в 1.1.0, подробности по раундам (`log`) копятся с 2.0.0. Файл 1.1.0
показывает карточки и график «поколений до финиша», а про остальное честно говорит,
что по старым раундам подробностей нет.
"""
import statistics

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

import main
import theme
from chart import Chart

# Замер автора (README): уровень «сложный», двенадцать случайных сидов - доехали десять,
# медиана 16.5 поколения до финиша.
BENCH_LEVEL = "сложный"
BENCH_FINISH_SHARE = 10 / 12
BENCH_MEDIAN_GENS = 16.5
GENERATORS = (("cppn", main.GEN_CPPN), ("model", main.GEN_MODEL))


def label(text="", name=None, wrap=False):
    lb = QLabel(text)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    return lb


def section(text):
    return label(text.upper(), "section", wrap=True)


def card():
    f = QFrame()
    f.setObjectName("card")
    return f


# --- счёт по записям раундов (чистые функции, без Qt) -------------------------------------

def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def entries(totals):
    log = totals.get("log") if isinstance(totals, dict) else None
    return [e for e in log if isinstance(e, dict)] if isinstance(log, list) else []


def _sub(entry, group, key):
    part = entry.get(group)
    value = part.get(key) if isinstance(part, dict) else None
    return value if _num(value) else None


def summary(rows):
    """Раундов, доехало, среднее поколений до финиша, лучшее время - по списку записей."""
    finished = [e for e in rows if e.get("finished") is True]
    gens = [e["gens"] for e in finished if _num(e.get("gens"))]
    times = [e["time"] for e in finished if _num(e.get("time"))]
    return {"rounds": len(rows), "finished": len(finished),
            "gens": sum(gens) / len(gens) if gens else None,
            "median_gens": statistics.median(gens) if gens else None,
            "time": min(times) if times else None}


def by_level(totals):
    rows = {name: [] for name in main.LEVEL_NAMES + (main.LEVEL_CUSTOM,)}
    for e in entries(totals):
        level = e.get("level")
        rows[level if level in rows else main.LEVEL_CUSTOM].append(e)
    return {name: summary(items) for name, items in rows.items()}


def by_generator(totals):
    out = {}
    for key, _ in GENERATORS:
        rows = [e for e in entries(totals) if e.get("generator") == key]
        info = summary(rows)
        for field, name, how in (("interest", "interest", statistics.median),
                                 ("length", "length", statistics.median),
                                 ("build_s", "build", statistics.fmean)):
            values = [v for v in (_sub(e, "track", field) for e in rows) if v is not None]
            info[name] = how(values) if values else None
        out[key] = info
    return out


def compare(totals):
    """Раунды на уровне замера: доля доехавших и медиана поколений - против замера."""
    info = summary([e for e in entries(totals) if e.get("level") == BENCH_LEVEL])
    share = info["finished"] / info["rounds"] if info["rounds"] else None
    return {"rounds": info["rounds"], "share": share, "median_gens": info["median_gens"],
            "share_ok": share is not None and share >= BENCH_FINISH_SHARE - 1e-9,
            "gens_ok": info["median_gens"] is not None and info["median_gens"] <= BENCH_MEDIAN_GENS}


def cars(totals):
    """Средние скорость, руль и масса машинок, которые доезжали и которые нет."""
    out = {}
    for key, want in (("finished", True), ("crashed", False)):
        rows = [e for e in entries(totals) if (e.get("finished") is True) == want]
        info = {"rounds": len(rows)}
        for field in ("speed", "steer", "mass"):
            values = [v for v in (_sub(e, "car", field) for e in rows) if v is not None]
            info[field] = sum(values) / len(values) if values else None
        out[key] = info
    return out


def curves(totals, history=None):
    """Точки графиков: обучение последнего раунда, поколений до финиша, время, прогресс."""
    if history and len(history) >= 2:
        best = [s.best for s in history]
        mean = [s.mean for s in history]
    else:
        curve = totals.get("last_curve") if isinstance(totals.get("last_curve"), dict) else {}
        best = [v for v in curve.get("best", []) if _num(v)] if isinstance(curve.get("best"), list) else []
        mean = [v for v in curve.get("mean", []) if _num(v)] if isinstance(curve.get("mean"), list) else []
    gens = totals.get("gens_to_finish") if isinstance(totals.get("gens_to_finish"), list) else []
    log = entries(totals)
    return {
        "training": [(i, v) for i, v in enumerate(best)],
        "training_mean": [(i, v) for i, v in enumerate(mean)],
        "gens": [(i, v) for i, v in enumerate((g for g in gens if _num(g)), start=1)],
        "time": [(e["n"], e["time"]) for e in log
                 if e.get("finished") is True and _num(e.get("n")) and _num(e.get("time"))],
        "progress": [(e["n"], e["progress"]) for e in log if _num(e.get("n")) and _num(e.get("progress"))],
    }


def _or_dash(v, form="{:.0f}"):
    return form.format(v) if v is not None else "-"


# --- экран ---------------------------------------------------------------------------------

class StatsScreen(QWidget):
    back = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("screen")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 22, 28, 16)
        head = QHBoxLayout()
        head.addWidget(label("Статистика", "screenTitle"))
        head.addStretch(1)
        self.back_button = QPushButton("К игре")
        self.back_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back_button.clicked.connect(self.back.emit)
        head.addWidget(self.back_button)
        outer.addLayout(head)
        outer.addSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("scrollBody")
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        self.scroll = scroll
        box = QVBoxLayout(body)
        box.setContentsMargins(0, 0, 8, 8)
        box.setSpacing(14)

        cards = QHBoxLayout()
        cards.setSpacing(12)
        self.values, self.notes = {}, {}
        for key, caption in (("rounds", "раундов сыграно"), ("finished", "доехали до финиша"),
                             ("generations", "поколений обучено"), ("best_time", "рекорд круга")):
            c = card()
            v = QVBoxLayout(c)
            v.setContentsMargins(16, 12, 16, 12)
            v.setSpacing(2)
            self.values[key] = label("0", "cardValue")
            v.addWidget(self.values[key])
            v.addWidget(label(caption, "muted", wrap=True))
            self.notes[key] = label("", "cardNote", wrap=True)
            v.addWidget(self.notes[key])
            v.addStretch(1)              # подписи всех карточек на одной высоте
            cards.addWidget(c, 1)
        box.addLayout(cards)

        self.level_cells = {}
        levels = card()
        grid = QGridLayout(levels)
        grid.setContentsMargins(18, 14, 18, 14)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(8)
        for col, text in enumerate(("Уровень", "Раундов", "Доехало", "Поколений\nдо финиша",
                                    "Лучшее\nвремя")):
            grid.addWidget(section(text), 0, col, Qt.AlignmentFlag.AlignBottom)
        grid.setColumnStretch(4, 1)
        for row, name in enumerate(main.LEVEL_NAMES + (main.LEVEL_CUSTOM,), start=1):
            grid.addWidget(label(name, "value"), row, 0)
            for col, key in enumerate(("rounds", "finished", "gens", "time"), start=1):
                cell = label("-")
                self.level_cells[name, key] = cell
                grid.addWidget(cell, row, col)
        self.levels_note = label("", "muted", wrap=True)
        grid.addWidget(self.levels_note, len(main.LEVEL_NAMES) + 2, 0, 1, 5)
        box.addWidget(levels)

        self.gen_cells = {}
        gens = card()
        grid = QGridLayout(gens)
        grid.setContentsMargins(18, 14, 18, 14)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(8)
        for col, text in enumerate(("Генератор трассы", "Трасс", "Доехало", "Интерес\n(медиана)",
                                    "Время\nна трассу", "Длина\n(медиана)")):
            grid.addWidget(section(text), 0, col, Qt.AlignmentFlag.AlignBottom)
        grid.setColumnStretch(5, 1)
        for row, (key, name) in enumerate(GENERATORS, start=1):
            grid.addWidget(label(name, "value"), row, 0)
            for col, field in enumerate(("rounds", "finished", "interest", "build", "length"), start=1):
                cell = label("-")
                self.gen_cells[key, field] = cell
                grid.addWidget(cell, row, col)
        box.addWidget(gens)

        bench = card()
        bv = QVBoxLayout(bench)
        bv.setContentsMargins(18, 14, 18, 14)
        bv.addWidget(section(f"Уровень «{BENCH_LEVEL}»: сравнение с замером автора на 12 сидах"))
        self.bench_label = label("", wrap=True)
        self.bench_label.setTextFormat(Qt.TextFormat.RichText)
        bv.addWidget(self.bench_label)
        box.addWidget(bench)

        garage = card()
        gv = QVBoxLayout(garage)
        gv.setContentsMargins(18, 14, 18, 14)
        gv.addWidget(section("Машинки: какие доезжают"))
        self.cars_label = label("", wrap=True)
        self.cars_label.setTextFormat(Qt.TextFormat.RichText)
        gv.addWidget(self.cars_label)
        box.addWidget(garage)

        charts = QGridLayout()
        charts.setSpacing(12)
        self.chart_training = Chart("Последний раунд: лучший и средний", theme.GOLD, "поколение")
        self.chart_gens = Chart("Поколений до финиша", theme.ACCENT, "раунд с финишем")
        self.chart_time = Chart("Время финиша", theme.GOOD, "раунд", unit=" с")
        self.chart_progress = Chart("Дальше всех проехал: доля трассы", "#de8c5a", "раунд", percent=True)
        self.charts = [self.chart_training, self.chart_gens, self.chart_time, self.chart_progress]
        for i, c in enumerate(self.charts):
            charts.addWidget(c, i // 2, i % 2)
        box.addLayout(charts)
        self.no_charts = label("Графиков пока нет: они появятся, когда наберётся хотя бы "
                               "два поколения или два раунда.", "muted", wrap=True)
        box.addWidget(self.no_charts)
        box.addStretch(1)

    # --- данные ---------------------------------------------------------------------------

    def refresh(self, totals, history=None):
        rounds, finished = totals.get("rounds", 0), totals.get("finished", 0)
        self.values["rounds"].setText(str(rounds))
        self.values["finished"].setText(str(finished))
        self.values["generations"].setText(str(totals.get("generations", 0)))
        best = totals.get("best_time")
        self.values["best_time"].setText(f"{best:.1f} с" if _num(best) else "-")
        self.notes["rounds"].setText("каждый - своя трасса или машинка")
        self.notes["finished"].setText(f"{finished / rounds * 100:.0f}% раундов" if rounds else "")
        done = [g for g in totals.get("gens_to_finish") or [] if _num(g)]
        avg = sum(done) / len(done) if done else None
        self.notes["generations"].setText(f"до финиша в среднем {avg:.1f}" if avg is not None else "")
        self.notes["best_time"].setText("лучший заезд до финиша" if _num(best) else "финиша ещё не было")

        for name, info in by_level(totals).items():
            self.level_cells[name, "rounds"].setText(str(info["rounds"]))
            self.level_cells[name, "finished"].setText(str(info["finished"]))
            self.level_cells[name, "gens"].setText(_or_dash(info["gens"], "{:.1f}"))
            self.level_cells[name, "time"].setText(_or_dash(info["time"], "{:.1f} с"))
        logged = len(entries(totals))
        older = max(0, rounds - logged)
        self.levels_note.setText(
            f"Раунды до версии 2.0.0 ({older}) есть только в общих счётчиках: подробности "
            f"о каждом раунде записываются с 2.0.0." if older else "")
        self.levels_note.setVisible(bool(older))

        for key, info in by_generator(totals).items():
            self.gen_cells[key, "rounds"].setText(str(info["rounds"]))
            self.gen_cells[key, "finished"].setText(str(info["finished"]))
            self.gen_cells[key, "interest"].setText(_or_dash(info["interest"], "{:.1f}"))
            build = info["build"]
            self.gen_cells[key, "build"].setText(
                "-" if build is None else f"{build * 1000:.0f} мс" if build < 1 else f"{build:.1f} с")
            self.gen_cells[key, "length"].setText(_or_dash(info["length"]))

        self.bench_label.setText(self._bench(compare(totals)))
        self.cars_label.setText(self._cars(cars(totals)))

        points = curves(totals, history)
        self.chart_training.set_points(points["training"], points["training_mean"])
        self.chart_gens.set_points(points["gens"])
        self.chart_time.set_points(points["time"])
        self.chart_progress.set_points(points["progress"])
        self.no_charts.setVisible(self.visible_charts() == 0)

    @staticmethod
    def _bench(info):
        muted = f"color:{theme.MUTED}"
        if not info["rounds"]:
            return (f"<span style='{muted}'>Раундов на этом уровне ещё нет. Замер автора: доезжают "
                    f"{BENCH_FINISH_SHARE * 100:.0f}% раундов, медиана {BENCH_MEDIAN_GENS:g} "
                    f"поколения до финиша.</span>")

        def mark(ok):
            return theme.GOOD if ok else theme.WARN

        share = f"{info['share'] * 100:.0f}%"
        gens = f"{info['median_gens']:g}" if info["median_gens"] is not None else "-"
        return (f"доехали до финиша: <b style='color:{mark(info['share_ok'])}'>{share}</b> "
                f"<span style='{muted}'>(замер: {BENCH_FINISH_SHARE * 100:.0f}%, раундов у вас "
                f"{info['rounds']})</span><br>"
                f"поколений до финиша, медиана: <b style='color:{mark(info['gens_ok'])}'>{gens}</b> "
                f"<span style='{muted}'>(замер: {BENCH_MEDIAN_GENS:g}, меньше - лучше)</span>")

    @staticmethod
    def _cars(info):
        muted = f"color:{theme.MUTED}"
        parts = []
        for key, name in (("finished", "доезжали"), ("crashed", "не доезжали")):
            row = info[key]
            if not row["rounds"]:
                continue
            parts.append(f"{name} ({row['rounds']}): скорость <b>{_or_dash(row['speed'])}</b>, "
                         f"руль <b>{_or_dash(row['steer'], '{:.2f}')}</b>, "
                         f"масса <b>{_or_dash(row['mass'], '{:.2f}')}</b>")
        if not parts:
            return f"<span style='{muted}'>Машинки запоминаются с версии 2.0.0: сыграйте раунд.</span>"
        parts.append(f"<span style='{muted}'>Длинная узкая машинка быстрее, но хуже входит в повороты: "
                     f"форма задаёт и скорость, и руль.</span>")
        return "<br>".join(parts)

    def visible_charts(self):
        return sum(not c.isHidden() for c in self.charts)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.back.emit()
            return
        super().keyPressEvent(event)
