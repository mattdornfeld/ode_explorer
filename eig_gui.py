#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from joblib import Parallel, delayed
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
  FigureCanvasQTAgg as FigureCanvas,
  NavigationToolbar2QT as NavigationToolbar)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QLabel, QGridLayout, QDockWidget, QApplication, 
  QMainWindow, QWidget, QLineEdit, QPushButton, QProgressBar)
from dynamics import flow, calc_jacobian, calc_fixed_points
import calc_parameters as cp
from constants import *
from IPython import embed

matplotlib.use("Qt5Agg")
plt.style.use('ggplot')

def calc_evs(p, r_front, r_back, s, nm):
    params = cp.calc_parameters(p, r_front, r_back, s.imag, s.real, nm)
    fp1, fp2 = calc_fixed_points(params)
    jac = calc_jacobian(fp1, params)
    evs, evecs = np.linalg.eig(jac)

    return evs

class MainWindow(QMainWindow):
  def __init__(self):
    QMainWindow.__init__(self)

    self.set_defaults()
    self.init_ui()
    self.update_plots()

  def closeEvent(self, event):
    plt.close('all')
    event.accept()

  def init_ui(self):
    #setup plot area
    self.fig, self.ax  = plt.subplots(1,2)
    self.canvas  = FigureCanvas(self.fig)
    self.setCentralWidget(self.canvas)
    
    #setup controls area
    dock = QDockWidget()
    self.addDockWidget(Qt.BottomDockWidgetArea, dock)
    self.controls = QWidget()
    self.controls_grid = QGridLayout(self.controls)
    self.controls_dict = {}
    self.mpl_toolbar = NavigationToolbar(self.canvas, self)
    self.controls_grid.addWidget( self.mpl_toolbar, 0, 2)

    self.add_text_box(update_fun=self.on_p_change,
      name='p', default_value=self.p, unit='pa', conversion=rho*c**2, 
      row=1, col=0)
    self.add_text_box(update_fun=self.on_dp_change,
      name='dp', default_value=self.dp, unit='pa', conversion=rho*c**2, 
      row=2, col=0)
    self.add_text_box(update_fun=self.on_r_front_change,
      name='r_front', default_value=self.r_front, unit='m', conversion=1,
       row=3, col=0)
    self.add_text_box(update_fun=self.on_r_back_change,
      name='r_back', default_value=self.r_back, unit='m', conversion=1,
       row=4, col=0)
    self.add_text_box(update_fun=self.on_nm_change,
      name='nm', default_value=self.nm, unit='modes', conversion=1, 
      row=1, col=2)
    self.add_text_box(update_fun=self.on_r_mouth_change,
      name='r_mouth', default_value=self.r_mouth, unit='m', conversion=1, 
      row=2, col=2)

    button = QPushButton('Update Plots')
    button.clicked.connect(self.on_push)
    self.controls_grid.addWidget(button, 3, 2)
    self.pbar = QProgressBar(self)
    self.controls_grid.addWidget(self.pbar, 0, 0)
    self.steps_label = QLabel()

    dock.setWidget( self.controls)


  def add_text_box(self, update_fun, name, default_value, unit, conversion, 
    row, col):
      qle = QLineEdit()
      qle.setText( str(default_value) )
      qle.textChanged[str].connect(update_fun)
      label = QLabel()
      self.controls_dict[name] = (qle, label, row, col, unit, conversion)
      label_text = (name + ' (' + str(default_value*conversion) + 
        ' ' + unit + ')')  
      label.setText(label_text)
      self.controls_grid.addWidget(qle, row, col)
      self.controls_grid.addWidget(label, row, col+1)

  def set_defaults(self):
      self.nm = 3
      self.r_mouth = 0.0003
      self.p = 0.015
      self.dp = 0.001
      self.r_front = 0.0026
      self.r_back = 0.001
      self.s = cp.calc_spatial_evs(r_mouth=self.r_mouth/L, nm=self.nm, 
        num_seeds=500, max_x=20, max_y=2)

  def update_plots(self):
      self.ax[0].cla()
      self.ax[1].cla()

      p_range = np.arange(0, self.p, self.dp)

      evs = np.array(
        Parallel(n_jobs=8)(delayed(calc_evs)(p, self.r_front, self.r_back, 
          self.s, self.nm) 
        for p in p_range) )

      self.ax[0].plot(rho*c**2*p_range, evs.real)
      self.ax[0].set_xlabel("pressure (pa)")
      self.ax[0].set_ylabel("Re(eigs)")

      self.ax[1].plot(rho*c**2*p_range, c/L/2/pi*abs(evs.imag))
      self.ax[1].set_xlabel("pressure (pa)")
      self.ax[1].set_ylabel("Im(eigs)")

      self.fig.canvas.draw()

      self.pbar.setValue(100)

  def get_label_text(self, value, name):
      unit = self.controls_dict[name][4]
      conversion = self.controls_dict[name][5]
      label_text = (name + ' (' + str(value * conversion) + 
      ' ' + unit + ')')

      return label_text
  
  def on_p_change(self, text):
      if len(text) != 0:
        try:
          self.p = float(text)
        except:
          raise ValueError('Input must be a float')

        name = 'p'
        label_text = self.get_label_text(self.p, name)
        self.controls_dict[name][1].setText(label_text)
        self.pbar.setValue(0)

  def on_dp_change(self, text):
    if len(text) != 0:
        try:
          self.dp = float(text)
        except:
          raise ValueError('Input must be a float')

        name = 'dp'
        label_text = self.get_label_text(self.dp, name)
        self.controls_dict[name][1].setText(label_text)
        self.pbar.setValue(0)

  def on_r_front_change(self, text):
      if len(text) != 0:
        try:
          self.r_front = float(text)
        except:
          raise ValueError('Input must be a float')

        name = 'r_front'
        label_text = self.get_label_text(self.r_front, name)
        self.controls_dict[name][1].setText(label_text)
        self.pbar.setValue(0)

  def on_r_back_change(self, text):
      if len(text) != 0:
        try:
          self.r_back = float(text)
        except:
          raise ValueError('Input must be a float')

        name = 'r_back'
        label_text = self.get_label_text(self.r_back, name)
        self.controls_dict[name][1].setText(label_text)
        self.pbar.setValue(0)

  def on_r_mouth_change(self, text):
      if len(text) != 0:
        try:
          self.r_mouth = float(text)
        except:
          raise ValueError('Input must be a float')

        name = 'r_mouth'
        label_text = self.get_label_text(self.r_mouth, name)
        self.controls_dict[name][1].setText(label_text)
        self.s = cp.calc_spatial_evs(r_mouth=self.r_mouth/L, nm=self.nm, 
          num_seeds=500, max_x=20, max_y=2)

        self.pbar.setValue(0)

  def on_nm_change(self, text):
    if len(text) != 0:
      try:
        self.nm = int(text)
      except:
        raise ValueError('Input must be an integer')

    name = 'nm'
    label_text = self.get_label_text(self.nm, name)
    self.controls_dict[name][1].setText(label_text)
    self.s = cp.calc_spatial_evs(r_mouth=self.r_mouth/L, nm=self.nm, 
      num_seeds=500, max_x=20, max_y=2)

    self.pbar.setValue(0)

  def on_push(self):
      self.update_plots()


if __name__ == "__main__":
  app = QApplication(sys.argv)
  main = MainWindow()
  main.show()
  sys.exit(app.exec_())