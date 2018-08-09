# -*- coding: utf-8 -*-

import sys

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QApplication, QWidget, QDialog, QMainWindow,
    QGridLayout, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QLineEdit, QSlider
)

from PyQt5.QtGui import QIcon, QPixmap, QValidator, QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt, pyqtSlot

from mechanisms import Scissor

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600


class FormWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.mechanism = None
        self.result = None
        self.movement_graph = None
        self.torque_graph = None
        self.safety_factor = 1
        self.init_ui()
        self.set_default_values()

    def init_ui(self):
        # input - mechanism parameters
        r1_label = QLabel('r1')
        self.r1_edit = QLineEdit()
        self.r1_edit.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.r1_edit.setValidator(QDoubleValidator(bottom=0))
        r1_unitlabel = QLabel('mm')

        p1_label = QLabel('p1')
        self.p1_edit = QLineEdit()
        self.p1_edit.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.p1_edit.setValidator(QDoubleValidator(bottom=0))
        p1_unitlabel = QLabel('mm')

        r2_label = QLabel('r2')
        self.r2_edit = QLineEdit()
        self.r2_edit.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.r2_edit.setValidator(QDoubleValidator(bottom=0))
        r2_unitlabel = QLabel('mm')

        p2_label = QLabel('p2')
        self.p2_edit = QLineEdit()
        self.p2_edit.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.p2_edit.setValidator(QDoubleValidator(bottom=0))
        p2_unitlabel = QLabel('mm')

        mass_label = QLabel('mass')
        self.mass_edit = QLineEdit()
        self.mass_edit.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.mass_edit.setValidator(QDoubleValidator(bottom=0))
        mass_unitlabel = QLabel('kg')

        parameters_grid = QGridLayout()
        parameters_grid.setSpacing(10)

        parameters_grid.addWidget(r1_label, 0, 0)
        parameters_grid.addWidget(self.r1_edit, 0, 1)
        parameters_grid.addWidget(r1_unitlabel, 0, 2)

        parameters_grid.addWidget(p1_label, 1, 0)
        parameters_grid.addWidget(self.p1_edit, 1, 1)
        parameters_grid.addWidget(p1_unitlabel, 1, 2)

        parameters_grid.addWidget(r2_label, 2, 0)
        parameters_grid.addWidget(self.r2_edit, 2, 1)
        parameters_grid.addWidget(r2_unitlabel, 2, 2)

        parameters_grid.addWidget(p2_label, 3, 0)
        parameters_grid.addWidget(self.p2_edit, 3, 1)
        parameters_grid.addWidget(p2_unitlabel, 3, 2)

        parameters_grid.addWidget(mass_label, 4, 0)
        parameters_grid.addWidget(self.mass_edit, 4, 1)
        parameters_grid.addWidget(mass_unitlabel, 4, 2)

        # input - theta12
        self.theta12_start_edit = QLineEdit()
        self.theta12_start_edit.setValidator(QIntValidator())
        self.theta12_start_edit.setAlignment(Qt.AlignRight | Qt.AlignBottom)

        self.theta12_end_edit = QLineEdit()
        self.theta12_end_edit.setValidator(QIntValidator())
        self.theta12_end_edit.setAlignment(Qt.AlignRight | Qt.AlignBottom)

        theta12Box = QHBoxLayout()
        theta12Box.addWidget(QLabel('Theta12'))
        theta12Box.addWidget(QLabel('from'))
        theta12Box.addWidget(self.theta12_start_edit)
        theta12Box.addWidget(QLabel('to'))
        theta12Box.addWidget(self.theta12_end_edit)
        theta12Box.addStretch(1)

        # input - safety factor
        self.sf_slider = QSlider(Qt.Horizontal)
        self.sf_slider.setRange(10, 100)
        self.sf_slider.setTickInterval(1)
        self.sf_slider.setSingleStep(1)
        self.sf_value_label = QLabel('{:.1f}'.format(self.safety_factor))
        self.sf_value_label.setFixedWidth(30)

        sfBox = QHBoxLayout()
        sfBox.addWidget(QLabel('Safety Factor'))
        sfBox.addWidget(self.sf_slider)
        sfBox.addWidget(self.sf_value_label)

        # output - results
        results_grid = QGridLayout()
        results_grid.setSpacing(10)

        torque_label = QLabel('Torque')
        self.torque_edit = QLineEdit()
        self.torque_edit.setReadOnly(True)
        self.torque_edit.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        torque_unitlabel = QLabel('N.m')

        positionX_label = QLabel('X')
        self.positionX_edit = QLineEdit()
        self.positionX_edit.setReadOnly(True)
        self.positionX_edit.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        positionX_unitlabel = QLabel('mm')

        positionZ_label = QLabel('Z')
        self.positionZ_edit = QLineEdit()
        self.positionZ_edit.setReadOnly(True)
        self.positionZ_edit.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        positionZ_unitlabel = QLabel('mm')

        results_grid.addWidget(torque_label, 0, 0)
        results_grid.addWidget(self.torque_edit, 0, 1)
        results_grid.addWidget(torque_unitlabel, 0, 2)

        results_grid.addWidget(self.positionX_edit, 1, 1)
        results_grid.addWidget(positionX_label, 1, 0)
        results_grid.addWidget(positionX_unitlabel, 1, 2)

        results_grid.addWidget(positionZ_label, 2, 0)
        results_grid.addWidget(self.positionZ_edit, 2, 1)
        results_grid.addWidget(positionZ_unitlabel, 2, 2)

        self.calculateButton = QPushButton("Calculate")
        self.plotButton = QPushButton("Plot")

        actionButtonBox = QHBoxLayout()
        actionButtonBox.addWidget(self.calculateButton)
        actionButtonBox.addWidget(self.plotButton)

        prBox = QVBoxLayout()
        prBox.addLayout(parameters_grid)
        prBox.addLayout(theta12Box)
        prBox.addLayout(actionButtonBox)
        prBox.addStretch(1)
        prBox.addLayout(sfBox)
        prBox.addLayout(results_grid)

        mechanismImageLabel = QLabel()
        mechanismImage = QPixmap()
        mechanismImage.load('./img/mechanism.jpeg')
        mechanismImageLabel.setPixmap(mechanismImage)

        doorImageLabel = QLabel()
        doorImage = QPixmap()
        doorImage.load('./img/door.png')
        doorImageLabel.setPixmap(doorImage)

        hbox = QHBoxLayout()
        hbox.addLayout(prBox)
        hbox.addWidget(mechanismImageLabel)
        hbox.addWidget(doorImageLabel)

        self.setLayout(hbox)

        with open("./css/style.css", "r") as f:
            self.setStyleSheet(f.read())

        x = int(SCREEN_WIDTH * 0.05)
        y = int(SCREEN_HEIGHT * 0.05)
        w = int(SCREEN_WIDTH * 0.75)
        h = int(SCREEN_HEIGHT * 0.4)
        self.setGeometry(x, y, w, h)

        self.setWindowTitle('Slider Crank')

        # SIGNAL & SLOT
        self.calculateButton.clicked.connect(self.calculate)
        self.plotButton.clicked.connect(self.plot)
        self.sf_slider.valueChanged.connect(self.refresh_result_ui)

    def set_default_values(self):
        self.theta12_start_edit.setText("45")
        self.theta12_end_edit.setText("-45")

        self.r1_edit.setText("185")
        self.p1_edit.setText("405")
        self.r2_edit.setText("185")
        self.p2_edit.setText("405")
        self.mass_edit.setText("6")

    def parse(self, element):
        textInput = element.text()
        validator = element.validator()
        state = validator.validate(textInput, 0)[0]

        if state == QValidator.Acceptable:
            return float(textInput)

        return None

    @pyqtSlot()
    def refresh_result_ui(self):
        self.safety_factor = self.sf_slider.value()/10.0
        self.sf_value_label.setText("{:.1f}".format(self.safety_factor))

        if self.result:
            # update ui
            torque = self.result['torque']

            torque_max = np.around(np.max(np.abs(torque)), decimals=2)*self.safety_factor

            H = self.result['H']
            H_max = np.max(np.abs(H))
            Z = int(H_max + 0.772) * 1000

            X = 500 - 329

            self.torque_edit.setText('{:.2f}'.format(torque_max))
            self.positionX_edit.setText('{:d}'.format(X))
            self.positionZ_edit.setText('{:d}'.format(Z))


    @pyqtSlot()
    def calculate(self):
        r1 = self.parse(self.r1_edit) / 1000
        p1 = self.parse(self.p1_edit) / 1000
        r2 = self.parse(self.r2_edit) / 1000
        p2 = self.parse(self.p2_edit) / 1000
        theta12_start = self.parse(self.theta12_start_edit)
        theta12_end = self.parse(self.theta12_end_edit)
        theta12_step = 0.5
        if theta12_start > theta12_end:
            theta12_step = -0.5

        mass = self.parse(self.mass_edit)

        if None in [r1, p1, r2, p2, mass]:
            return

        self.mechanism = Scissor(first_link=r1, second_link=r2, first_link_total=p1, second_link_total=p2, mass=mass)

        theta12 = np.deg2rad(np.arange(theta12_start, theta12_end, theta12_step))
        result = self.mechanism.solve(theta12)

        # update fields
        self.result = result

        # update results
        self.refresh_result_ui()

        print(r1, p1, r2, p2, mass)
        print(self.result)
        print("calculated..")

    @pyqtSlot()
    def plot(self):
        self.plot_angles()
        self.plot_torque()

    def plot_angles(self):
        self.movement_graph = MovementGraph(data=self.result, parent=self)
        self.movement_graph.show()
        self.movement_graph.raise_()
        self.movement_graph.activateWindow()

    def plot_torque(self):
        self.torque_graph = TorqueGraph(data=self.result, parent=self)
        self.torque_graph.show()
        self.torque_graph.raise_()
        self.torque_graph.activateWindow()


class FigureView(QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)

        self.figure, self.axes = self.createFigureAxes()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)

        # VIEW
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlags(self.windowFlags() | Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint)

        # LAYOUT
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # PLOT
        self.plot(data)

    def createFigureAxes(self):
        figure, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=80)
        return figure, axes

    def plot(self, data):
        pass


class MovementGraph(FigureView):
    def createFigureAxes(self):
        figure, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 9), dpi=80)
        return figure, axes

    def plot(self, data):
        theta12 = data['theta12']
        theta13 = data['theta13']
        s14 = data['s14']
        H = data['H']

        ax1, ax2, ax3 = self.axes

        ax1.plot(np.rad2deg(theta12), np.rad2deg(theta13), color="blue", linewidth=2.5, label=r'$\theta_{13}$')
        ax2.plot(np.rad2deg(theta12), s14 * 1000, color="red", linewidth=2.5, label=r'$s_{14}$')
        ax3.plot(np.rad2deg(theta12), (H - 0.250) * -1 * 1000, color="blue", linewidth=2.5, label=r'$H$')

        ax1.grid(True)
        ax1.set_ylabel(r'$\theta_{13}$ [°]')

        ax2.grid(True)
        ax2.set_ylabel("S14 [mm]")

        ax3.grid(True)
        ax3.set_ylabel("H [mm]")
        ax3.set_xlabel(r'$\theta_{12}$ [°]')

        x = int(SCREEN_WIDTH * 0.5)
        y = int(SCREEN_HEIGHT * 0.05)
        w = int(SCREEN_WIDTH * 0.45)
        h = int(SCREEN_HEIGHT * 0.9)
        self.setGeometry(x, y, w, h)
        self.canvas.draw()


class TorqueGraph(FigureView):
    def createFigureAxes(self):
        figure, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=80)
        return figure, (axes,)

    def plot(self, data):
        theta12 = data['theta12']
        torque = data['torque']

        ax1 = self.axes[0]

        ax1.plot(np.rad2deg(theta12), torque, color="r", linewidth=3, label='torque')
        ax1.grid(True)
        ax1.set_xlabel(r'$\theta_{12}$ [°]')
        ax1.set_ylabel("Tork [N*m]")

        x = int(SCREEN_WIDTH * 0.05)
        y = int(SCREEN_HEIGHT * 0.5)
        w = int(SCREEN_WIDTH * 0.4)
        h = int(SCREEN_HEIGHT * 0.4)
        self.setGeometry(x, y, w, h)
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    screen = app.desktop().screenGeometry()
    SCREEN_WIDTH, SCREEN_HEIGHT = screen.width(), screen.height()
    form = FormWidget()
    form.show()
    sys.exit(app.exec_())
