# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import PyQt5
from PyQt5.QtWidgets import (
    QApplication, QWidget, QDialog, QMainWindow,
    QGridLayout, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QLineEdit, QSlider, QSizePolicy
)

from PyQt5.QtGui import QIcon, QPixmap, QValidator, QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt, pyqtSlot

from mechanisms import Scissor

# bundle vs source path
frozen = False
if getattr(sys, 'frozen', False):
    # we are running in a bundle
    frozen = True
    bundle_dir = sys._MEIPASS
else:
    # we are running in a normal Python environment
    bundle_dir = os.path.dirname(os.path.abspath(__file__))


def get_abs_path(relative_path):
    return os.path.abspath(os.path.join(bundle_dir, relative_path))


# GLOBAL VARIABLES
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600

stopper1 = (0.214, 0)
stopper2 = (0.366, 0)


class FormWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.mechanism = None
        self.result = None
        self.movement_graph = None
        self.torque_graph = None
        self.animation_graph = None
        self.safety_factor = 1.5
        self.init_ui()
        self.set_default_values()

    def init_ui(self):
        # input - mechanism parameters
        r2_label = QLabel('r1')
        self.r2_edit = QLineEdit()
        self.r2_edit.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.r2_edit.setValidator(QDoubleValidator(bottom=0))
        r2_unitlabel = QLabel('mm')
        self.r2_button = QPushButton('?')

        p2_label = QLabel('p1')
        self.p2_edit = QLineEdit()
        self.p2_edit.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.p2_edit.setValidator(QDoubleValidator(bottom=0))
        p2_unitlabel = QLabel('mm')
        self.p2_button = QPushButton('?')

        r3_label = QLabel('r2')
        self.r3_edit = QLineEdit()
        self.r3_edit.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.r3_edit.setValidator(QDoubleValidator(bottom=0))
        r3_unitlabel = QLabel('mm')
        self.r3_button = QPushButton('?')

        p3_label = QLabel('p2')
        self.p3_edit = QLineEdit()
        self.p3_edit.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.p3_edit.setValidator(QDoubleValidator(bottom=0))
        p3_unitlabel = QLabel('mm')
        self.p3_button = QPushButton('?')

        mass_label = QLabel('mass')
        self.mass_edit = QLineEdit()
        self.mass_edit.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.mass_edit.setValidator(QDoubleValidator(bottom=0))
        mass_unitlabel = QLabel('kg')

        parameters_grid = QGridLayout()
        parameters_grid.setSpacing(10)

        parameters_grid.addWidget(r2_label, 0, 0)
        parameters_grid.addWidget(self.r2_edit, 0, 1)
        parameters_grid.addWidget(r2_unitlabel, 0, 2)
        parameters_grid.addWidget(self.r2_button, 0, 3)

        parameters_grid.addWidget(p2_label, 1, 0)
        parameters_grid.addWidget(self.p2_edit, 1, 1)
        parameters_grid.addWidget(p2_unitlabel, 1, 2)
        parameters_grid.addWidget(self.p2_button, 1, 3)

        parameters_grid.addWidget(r3_label, 2, 0)
        parameters_grid.addWidget(self.r3_edit, 2, 1)
        parameters_grid.addWidget(r3_unitlabel, 2, 2)
        parameters_grid.addWidget(self.r3_button, 2, 3)

        parameters_grid.addWidget(p3_label, 3, 0)
        parameters_grid.addWidget(self.p3_edit, 3, 1)
        parameters_grid.addWidget(p3_unitlabel, 3, 2)
        parameters_grid.addWidget(self.p3_button, 3, 3)

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
        self.sf_slider.setSliderPosition(self.safety_factor * 10)
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

        glass_height_label = QLabel('Window Opening')
        self.glass_height_edit = QLineEdit()
        self.glass_height_edit.setReadOnly(True)
        self.glass_height_edit.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        glass_height_unitlabel = QLabel('mm')

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

        results_grid.addWidget(glass_height_label, 1, 0)
        results_grid.addWidget(self.glass_height_edit, 1, 1)
        results_grid.addWidget(glass_height_unitlabel, 1, 2)

        results_grid.addWidget(positionX_label, 2, 0)
        results_grid.addWidget(self.positionX_edit, 2, 1)
        results_grid.addWidget(positionX_unitlabel, 2, 2)

        results_grid.addWidget(positionZ_label, 3, 0)
        results_grid.addWidget(self.positionZ_edit, 3, 1)
        results_grid.addWidget(positionZ_unitlabel, 3, 2)

        self.calculateButton = QPushButton("Calculate")
        self.plotButton = QPushButton("Plot")
        self.animateButton = QPushButton("Animate")

        actionButtonBox = QVBoxLayout()
        actionButtonBox.addWidget(self.calculateButton)
        actionButtonBox.addWidget(self.plotButton)
        actionButtonBox.addWidget(self.animateButton)

        prBox = QVBoxLayout()
        prBox.addLayout(parameters_grid)
        prBox.addLayout(theta12Box)
        prBox.addLayout(actionButtonBox)
        prBox.addStretch(1)
        prBox.addLayout(sfBox)
        prBox.addLayout(results_grid)

        mechanismImageLabel = QLabel()
        mechanismImage = QPixmap()
        mechanismImage.load(get_abs_path('./img/mechanism.jpeg'))
        mechanismImageLabel.setPixmap(mechanismImage)

        doorImageLabel = QLabel()
        doorImage = QPixmap()
        doorImage.load(get_abs_path('./img/door.png'))
        doorImageLabel.setPixmap(doorImage)

        hbox = QHBoxLayout()
        hbox.addLayout(prBox)
        hbox.addWidget(mechanismImageLabel)
        hbox.addWidget(doorImageLabel)

        self.setLayout(hbox)

        x = int(SCREEN_WIDTH * 0.05)
        y = int(SCREEN_HEIGHT * 0.05)
        w = int(SCREEN_WIDTH * 0.6)
        h = int(SCREEN_HEIGHT * 0.4)
        self.setGeometry(x, y, w, h)

        self.setWindowTitle('Slider Crank')

        # SIGNAL & SLOT
        self.r2_edit.textChanged.connect(self.hide)
        self.p2_edit.textChanged.connect(self.hide)
        self.r3_edit.textChanged.connect(self.hide)
        self.p3_edit.textChanged.connect(self.hide)
        self.mass_edit.textChanged.connect(self.hide)
        self.theta12_start_edit.textChanged.connect(self.hide)
        self.theta12_end_edit.textChanged.connect(self.hide)

        self.sf_slider.valueChanged.connect(self.refresh_safety_factor)

        self.calculateButton.clicked.connect(self.calculate)
        self.plotButton.clicked.connect(self.plot)
        self.animateButton.clicked.connect(self.animate)

        self.r2_button.clicked.connect(self.showImage)
        self.p2_button.clicked.connect(self.showImage)
        self.r3_button.clicked.connect(self.showImage)
        self.p3_button.clicked.connect(self.showImage)

    def set_default_values(self):
        self.theta12_start_edit.setText("45")
        self.theta12_end_edit.setText("-45")

        self.r2_edit.setText("185")
        self.p2_edit.setText("405")
        self.r3_edit.setText("185")
        self.p3_edit.setText("405")
        self.mass_edit.setText("6")

    def parse(self, element):
        textInput = element.text()
        validator = element.validator()
        state = validator.validate(textInput, 0)[0]

        if state == QValidator.Acceptable:
            return float(textInput)

        return None

    @pyqtSlot()
    def showImage(self):
        sending_button = self.sender()
        img_src = ""
        if sending_button is self.r2_button:
            img_src = get_abs_path("./img/r2.png")
        elif sending_button is self.p2_button:
            img_src = get_abs_path("./img/p2.png")
        elif sending_button is self.r3_button:
            img_src = get_abs_path("./img/r3.png")
        elif sending_button is self.p3_button:
            img_src = get_abs_path("./img/p3.png")

        img_pop = PopupImage(img_src=img_src, parent=self)
        img_pop.show()
        img_pop.raise_()
        img_pop.activateWindow()

    @pyqtSlot()
    def hide(self):
        self.torque_edit.setText('')
        self.positionX_edit.setText('')
        self.positionZ_edit.setText('')
        self.plotButton.setVisible(False)
        self.animateButton.setVisible(False)

    @pyqtSlot()
    def calculate(self):
        r2 = self.parse(self.r2_edit) / 1000
        p2 = self.parse(self.p2_edit) / 1000
        r3 = self.parse(self.r3_edit) / 1000
        p3 = self.parse(self.p3_edit) / 1000
        theta12_start = self.parse(self.theta12_start_edit)
        theta12_end = self.parse(self.theta12_end_edit)
        theta12_step = 0.5
        if theta12_start > theta12_end:
            theta12_step = -0.5

        mass = self.parse(self.mass_edit)

        if None in [r2, p2, r3, p3, mass, theta12_start, theta12_end]:
            return

        self.mechanism = Scissor(first_link=r2, second_link=r3, first_link_total=p2, second_link_total=p3, mass=mass)

        theta12 = np.deg2rad(np.arange(theta12_start, theta12_end + theta12_step, theta12_step))
        result = self.mechanism.solve(theta12)

        # update fields
        self.result = result

        # update results
        self.refresh_result_ui()

    @pyqtSlot()
    def refresh_result_ui(self):
        self.safety_factor = self.sf_slider.value() / 10.0

        if self.result:
            # update ui
            torque = self.result['torque']

            torque_max = np.around(np.max(np.abs(torque)), decimals=2) * self.safety_factor

            H = self.result['H']
            glass_height = int(np.abs(H.max() - H.min()) * 1000)

            H_max = np.max(np.abs(H))
            Z = int(H_max + 0.772) * 1000
            X = 500 - 329

            self.torque_edit.setText('{:.2f}'.format(torque_max))
            self.glass_height_edit.setText('{:d}'.format(glass_height))
            self.positionX_edit.setText('{:d}'.format(X))
            self.positionZ_edit.setText('{:d}'.format(Z))

            self.plotButton.setVisible(True)
            self.animateButton.setVisible(True)

    @pyqtSlot()
    def refresh_safety_factor(self):
        self.safety_factor = self.sf_slider.value() / 10.0
        self.sf_value_label.setText("{:.1f}".format(self.safety_factor))

        if self.result:
            # update ui
            torque = self.result['torque']

            torque_max = np.around(np.max(np.abs(torque)), decimals=2) * self.safety_factor
            self.torque_edit.setText('{:.2f}'.format(torque_max))

    @pyqtSlot()
    def plot(self):
        if not self.result:
            return

        theta12_start = self.parse(self.theta12_start_edit)
        theta12_end = self.parse(self.theta12_end_edit)
        self.result['xlim'] = (theta12_start, theta12_end)

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

    @pyqtSlot()
    def animate(self):
        if not self.result:
            return

        self.animation_graph = MechAnimation(data=self.result, parent=self)
        self.animation_graph.show()
        self.animation_graph.raise_()
        self.animation_graph.activateWindow()


class PopupImage(QDialog):
    def __init__(self, img_src, parent=None):
        super().__init__(parent)

        # VIEW
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlags(self.windowFlags() | Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint)

        # Image
        imageLabel = QLabel()
        image = QPixmap()
        image.load(img_src)
        imageLabel.setPixmap(image)

        # LAYOUT
        layout = QVBoxLayout()
        layout.addWidget(imageLabel)
        self.setLayout(layout)
        self.show()


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

        ax1.plot(np.rad2deg(theta12), np.rad2deg(theta13), color="red", linewidth=2.5, label=r'$\theta_{13}$')
        ax2.plot(np.rad2deg(theta12), s14 * 1000, color="green", linewidth=2.5, label=r'$s_{14}$')
        ax3.plot(np.rad2deg(theta12), H * 1000, color="blue", linewidth=2.5, label=r'$H$')

        ax1.grid(True)
        ax1.set_ylabel(r'$\theta_{13}$ [°]')

        ax2.grid(True)
        ax2.set_ylabel(r'$s_{14}$ [mm]')

        ax3.grid(True)
        ax3.set_ylabel("H [mm]")
        ax3.set_xlabel(r'$\theta_{12}$ [°]')
        ax3.set_xlim(data['xlim'])

        x = int(SCREEN_WIDTH * 0.5)
        y = int(SCREEN_HEIGHT * 0.05)
        w = int(SCREEN_WIDTH * 0.45)
        h = int(SCREEN_HEIGHT * 0.9)
        self.setGeometry(x, y, w, h)
        self.canvas.draw()


class TorqueGraph(FigureView):
    def createFigureAxes(self):
        figure, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=80)
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
        y = int(SCREEN_HEIGHT * 0.50)
        w = int(SCREEN_WIDTH * 0.4)
        h = int(SCREEN_HEIGHT * 0.4)
        self.setGeometry(x, y, w, h)
        self.canvas.draw()


class MyFuncAnimation(animation.FuncAnimation):
    """
    Unfortunately, it seems that the _blit_clear method of the Animation
    class contains an error in several matplotlib verions
    That's why, I fork it here and insert the latest git version of
    the function.
    """

    def _blit_clear(self, artists, bg_cache):
        # Get a list of the axes that need clearing from the artists that
        # have been drawn. Grab the appropriate saved background from the
        # cache and restore.
        axes = set(a.axes for a in artists)
        for a in axes:
            if a in bg_cache:  # this is the previously missing line
                a.figure.canvas.restore_region(bg_cache[a])


class MechAnimation(QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)

        fig = plt.figure(figsize=(8, 8), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)  # create an axis

        # axis props
        ax.set_aspect('equal', 'box')
        ax.set_xlim((-0.6 * 1000, 0.9 * 1000))
        ax.set_ylim((-0.4 * 1000, 1.1 * 1000))

        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')

        ax.tick_params(
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,  # labels along the bottom edge are off
            labelleft=False)

        self.fig = fig
        self.ax = ax
        self.canvas = canvas
        self.data = data

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # geometry
        x = int(SCREEN_WIDTH * 0.2)
        y = int(SCREEN_HEIGHT * 0.2)
        w = int(SCREEN_WIDTH * 0.4)
        h = int(SCREEN_HEIGHT * 0.4)
        self.setGeometry(x, y, w, h)

        #
        self.start()

    @pyqtSlot()
    def start(self):
        theta12 = self.data['theta12']
        half_cycle = np.arange(len(theta12))
        full_cycle = np.concatenate([half_cycle, half_cycle[::-1]])

        self.animation = MyFuncAnimation(self.fig, self.animate, init_func=self.init,
                                         frames=full_cycle, interval=5, repeat=True, blit=True)
        self.canvas.draw()

    def init(self):
        ax = self.ax

        # static elements
        pointA = ax.text(-0.05 * 1000, -0.01 * 1000, 'M', color='r')

        ax.plot([stopper1[0] * 1000 * 0.98, stopper2[0] * 1000 * 1.02],
                [stopper1[1] * 1000, stopper2[1] * 1000],
                color='w', lw=10, path_effects=[pe.Stroke(linewidth=15, foreground='gray'), pe.Normal()])

        pointStoppers, = ax.plot([stopper1[0] * 1000, stopper2[0] * 1000],
                                 [stopper1[1] * 1000, stopper2[1] * 1000],
                                 color='w', alpha=1, marker='h', mfc='r', ms=10)

        # dynamic elements
        line2, = ax.plot([], [], marker='o', c='k', lw=3, ms=5)
        line3, = ax.plot([], [], marker='o', c='k', lw=3, ms=5)
        pointB, = ax.plot([], [], marker='o', c='y', lw=3, ms=10, mfc='none', markeredgewidth=5)
        pointC, = ax.plot([], [], marker='o', c='y', lw=3, ms=10, mfc='none', markeredgewidth=5)
        pointD, = ax.plot([], [], marker='o', c='y', lw=3, ms=10, mfc='none', markeredgewidth=5)

        glass = Rectangle((0, 0), width=800, height=700, fc='blue', alpha=0.1)
        ax.add_patch(glass)

        self.line2 = line2
        self.line3 = line3
        self.pointB = pointB
        self.pointC = pointC
        self.pointD = pointD
        self.glass = glass

        return line2, line3, pointB, pointC, pointD, glass

    def animate(self, i):
        r2 = self.data['r2']
        p2 = self.data['p2']
        r3 = self.data['r3']
        p3 = self.data['p3']

        th12 = self.data['theta12'][i]
        th13 = self.data['theta13'][i]
        s = self.data['s14'][i]

        coordinateD = (s + p3 * np.cos(th13 - np.pi)), (p3 * np.sin(th13 - np.pi))
        coordinateGlass = coordinateD[0] - 0.2, coordinateD[1]

        self.line2.set_data([0 * 1000, r2 * np.cos(th12) * 1000, p2 * np.cos(th12) * 1000],
                            [0 * 1000, r2 * np.sin(th12) * 1000, p2 * np.sin(th12) * 1000])

        self.line3.set_data([s * 1000, (s + p3 * np.cos(th13 - np.pi)) * 1000],
                            [0 * 1000, (p3 * np.sin(th13 - np.pi)) * 1000])

        self.pointB.set_data([s * 1000],
                             [0 * 1000])

        self.pointC.set_data([p2 * np.cos(th12) * 1000],
                             [p2 * np.sin(th12) * 1000])

        self.pointD.set_data([coordinateD[0] * 1000],
                             [coordinateD[1] * 1000])

        self.glass.set_xy((coordinateGlass[0] * 1000, coordinateGlass[1] * 1000))

        self.canvas.draw()

        return self.line2, self.line3, self.pointB, self.pointC, self.pointD, self.glass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    screen = app.desktop().screenGeometry()
    SCREEN_WIDTH, SCREEN_HEIGHT = screen.width(), screen.height()
    form = FormWidget()
    form.show()
    sys.exit(app.exec_())
