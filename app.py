import sys
from PyQt5.QtCore import QDateTime, Qt
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDial,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollBar,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyleFactory,
    QTabWidget,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class DemoWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YourApp")
        self.resize(860, 640)

        central_widget = QWidget()
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        root_layout.addLayout(self._build_top_bar())

        group_splitter = QSplitter(Qt.Horizontal)
        group_splitter.addWidget(self._build_group1())
        group_splitter.addWidget(self._build_group2())
        group_splitter.setSizes([360, 460])
        root_layout.addWidget(group_splitter)

        root_layout.addLayout(self._build_bottom_area())

        self.progress = QProgressBar()
        self.progress.setValue(27)
        root_layout.addWidget(self.progress)

        self.setCentralWidget(central_widget)

    def _build_top_bar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Style:"))

        self.style_combo = QComboBox()
        self.style_combo.addItems(QStyleFactory.keys())
        self.style_combo.setCurrentText("Fusion")
        self.style_combo.currentTextChanged.connect(self._change_style)
        row.addWidget(self.style_combo)

        row.addStretch(1)

        self.standard_palette = QCheckBox("Use style's standard palette")
        self.standard_palette.setChecked(True)
        self.standard_palette.toggled.connect(self._toggle_palette)
        row.addWidget(self.standard_palette)

        self.disable_widgets = QCheckBox("Disable widgets")
        self.disable_widgets.toggled.connect(self._toggle_widget_state)
        row.addWidget(self.disable_widgets)
        return row

    def _build_group1(self) -> QGroupBox:
        box = QGroupBox("Group 1")
        layout = QVBoxLayout(box)
        layout.addWidget(QRadioButton("Radio button 1"))
        layout.addWidget(QRadioButton("Radio button 2"))
        layout.addWidget(QRadioButton("Radio button 3"))

        tri_state = QCheckBox("Tri-state check box")
        tri_state.setTristate(True)
        tri_state.setCheckState(Qt.PartiallyChecked)
        layout.addWidget(tri_state)
        layout.addStretch(1)
        return box

    def _build_group2(self) -> QGroupBox:
        box = QGroupBox("Group 2")
        layout = QVBoxLayout(box)

        default_button = QPushButton("Default Push Button")
        default_button.setDefault(True)
        layout.addWidget(default_button)

        toggle_button = QPushButton("Toggle Push Button")
        toggle_button.setCheckable(True)
        layout.addWidget(toggle_button)

        flat_button = QPushButton("Flat Push Button")
        flat_button.setFlat(True)
        layout.addWidget(flat_button)

        layout.addStretch(1)
        return box

    def _build_bottom_area(self) -> QHBoxLayout:
        row = QHBoxLayout()

        tabs = QTabWidget()
        tabs.addTab(self._build_table_tab(), "Table")
        tabs.addTab(self._build_text_tab(), "Text Edit")
        row.addWidget(tabs, 3)

        row.addWidget(self._build_group3(), 2)
        return row

    def _build_table_tab(self) -> QWidget:
        table_widget = QWidget()
        layout = QVBoxLayout(table_widget)

        table = QTableView()
        model = QStandardItemModel(4, 2)
        model.setHorizontalHeaderLabels(["1", "2"])
        for r in range(4):
            for c in range(2):
                model.setItem(r, c, QStandardItem(""))
        model.setVerticalHeaderLabels([str(i) for i in range(1, 5)])
        table.setModel(model)
        table.setMinimumHeight(210)

        layout.addWidget(table)

        scrollbar = QScrollBar(Qt.Horizontal)
        layout.addWidget(scrollbar)
        return table_widget

    def _build_text_tab(self) -> QWidget:
        text_widget = QWidget()
        layout = QVBoxLayout(text_widget)
        editor = QTextEdit()
        editor.setPlaceholderText("Buraya örnek metin yazabilirsiniz...")
        layout.addWidget(editor)
        return text_widget

    def _build_group3(self) -> QGroupBox:
        box = QGroupBox("Group 3")
        layout = QGridLayout(box)

        password = QLineEdit()
        password.setEchoMode(QLineEdit.Password)
        password.setText("123456")
        layout.addWidget(password, 0, 0, 1, 2)

        spin = QSpinBox()
        spin.setValue(50)
        layout.addWidget(spin, 1, 0, 1, 2)

        date_time = QDateTimeEdit(QDateTime.currentDateTime())
        date_time.setDisplayFormat("dd.MM.yyyy HH:mm")
        layout.addWidget(date_time, 2, 0, 1, 2)

        slider = QSlider(Qt.Horizontal)
        slider.setValue(30)
        layout.addWidget(slider, 3, 0)

        dial = QDial()
        dial.setValue(45)
        layout.addWidget(dial, 3, 1)

        second_scroll = QScrollBar(Qt.Horizontal)
        second_scroll.setValue(35)
        layout.addWidget(second_scroll, 4, 0)

        layout.setRowStretch(5, 1)
        return box

    def _toggle_widget_state(self, disabled: bool) -> None:
        for child in self.centralWidget().findChildren(QWidget):
            if child is not self.disable_widgets:
                child.setDisabled(disabled)
        self.disable_widgets.setDisabled(False)

    def _change_style(self, style_name: str) -> None:
        QApplication.setStyle(style_name)
        self._toggle_palette(self.standard_palette.isChecked())

    def _toggle_palette(self, use_standard_palette: bool) -> None:
        if use_standard_palette:
            QApplication.setPalette(QApplication.style().standardPalette())
        else:
            QApplication.setPalette(self.style().standardPalette())


def main() -> None:
    app = QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
