#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from math import pi
import numpy as np
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_qt5agg
from matplotlib.backends.backend_qt5agg import (
  NavigationToolbar2QT as NavigationToolbar)
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QLabel, QSlider, QGridLayout,QDockWidget, 
  QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, 
  QWidget, QLineEdit, QPushButton, QProgressBar)
from dynamics import odeint, flow, calc_jacobian, calc_fixed_points, by_term
import calc_parameters as cp
from constants import *
from IPython import embed

matplotlib.use("Qt5Agg")
plt.style.use('ggplot')

class MainWindow(QMainWindow):
  
  def __init__(self):
    QMainWindow.__init__(self)

    self.set_defaults()
    self.init_ui()
    self.run_simulation()
    self.update_plots()

  def closeEvent(self, event):
    plt.close('all')
    event.accept()

  def init_ui(self):
    #setup plot area
    self.fig1, self.ax1  = plt.subplots(1,3)
    self.fig2, self.ax2 = plt.subplots(4,2)
    self.canvas  = matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg(self.fig1)
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
    self.add_text_box(update_fun=self.on_r_folds_change,
      name='r_folds', default_value=self.r_folds, unit='m', conversion=1,
       row=2, col=0)
    self.add_text_box(update_fun=self.on_omega_d_change,
      name='omega_d', default_value=self.omega_d, unit='1/s', conversion=c/L/2/pi,
       row=3, col=0)
    self.add_text_box(update_fun=self.on_T_change,
      name='T', default_value=self.T, unit='s', conversion=L/c, 
      row=1, col=2)
    self.add_text_box(update_fun=self.on_fs_change, 
      name='fs', default_value=self.fs, unit='Hz', conversion=c/L,
       row=2, col=2)
    self.add_text_box(update_fun=self.on_nm_change,
      name='nm', default_value=self.nm, unit='modes', conversion=1, 
      row=3, col=2)
    self.add_text_box(update_fun=self.on_t1_change,
      name='t1', default_value=self.t1, unit='steps', conversion=1,
      row=1, col=4)
    self.add_text_box(update_fun=self.on_t2_change,
      name='t2', default_value=self.t2, unit='steps', conversion=1, 
      row=2, col=4)
    self.add_text_box(update_fun=self.on_Gamma_change,
      name='Gamma', default_value=self.Gamma, unit='', conversion=1, 
      row=4, col=2)
    self.add_text_box(update_fun=self.on_r_mouth_change,
      name='r_mouth', default_value=self.r_mouth, unit='m', conversion=1, 
      row=4, col=0)

    button = QPushButton('Run Simulation')
    button.clicked.connect(self.on_push)
    self.controls_grid.addWidget(button, 3, 4)

    page_one_button = QPushButton('Page 1')
    page_one_button.clicked.connect(self.on_page_one)
    self.controls_grid.addWidget(page_one_button, 5, 0)

    page_two_button = QPushButton('Page 2')
    page_two_button.clicked.connect(self.on_page_two)
    self.controls_grid.addWidget(page_two_button, 5, 1)

    self.pbar = QProgressBar(self)
    self.controls_grid.addWidget(self.pbar, 0, 0)
    self.steps_label = QLabel()
    self.controls_grid.addWidget(self.steps_label, 0, 1)
    self.steps_label.setText('n_steps=' + str(len(self.t))) 

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
      self.page = 1

      self.nm = 3
      self.r_mouth = 0.001
      self.omega_d = pi / 2
      self.Gamma = 1e-4

      self.p = 0.015
      self.r_folds = 0.002
      self.r_back = 0.002

      self.fs = 3. #time sampling frequency
      self.T = 500. #total time
      self.t = np.arange(0, self.T, 1/self.fs)
      self.t1 = 0 
      self.t2 = len(self.t)
      self.s = cp.calc_spatial_evs(r_mouth=self.r_mouth/L, nm=self.nm, 
          num_seeds=500, max_x=20, max_y=2)
      self.params = cp.calc_parameters(self.p, self.r_folds, self.r_folds, 
        self.omega_d, self.Gamma, self.s.imag, self.s.real, self.nm)
      fp1, fp2 = calc_fixed_points(self.params)
      #self.q0 = fp1 + 1e-6 * abs(np.random.rand(2*self.nm+1))
  
  def flow_progress(self, q, t, params):
      if t/self.T*100 < 100:
        value = t/self.T*100
      else:
        value = 100

      self.pbar.setValue(value)

      return flow(q, t, params)
      
  def run_simulation(self):
      #self.q = odeint( self.flow_progress, self.q0, self.params, self.fs, self.T )
      fp1, fp2 = calc_fixed_points(self.params)
      self.q0 = np.zeros(2*self.nm+3)
      self.q0[self.nm+2] = 1e-8
      #self.q0 = fp1 + 1e-6 * abs(np.random.rand(2*self.nm+1))
      self.q = integrate.odeint( self.flow_progress, self.q0, self.t, args = (self.params,) , mxstep = 0 )

  def on_page_one(self):
      self.page = 1
      self.canvas  = matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg(self.fig1)
      self.setCentralWidget(self.canvas)
      self.mpl_toolbar = NavigationToolbar(self.canvas, self)
      self.controls_grid.addWidget( self.mpl_toolbar, 0, 2)
      self.update_plots()

  def on_page_two(self):
      self.page = 2
      self.canvas  = matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg(self.fig2)
      self.setCentralWidget(self.canvas)
      self.mpl_toolbar = NavigationToolbar(self.canvas, self)
      self.controls_grid.addWidget( self.mpl_toolbar, 0, 2)
      self.update_plots()

  def plot_page_one(self):
      [ax.cla() for ax in self.ax1.flatten()]

      #plot driving velocity
      t = self.t[self.t1:self.t2] * L / c
      a = np.real(self.s * np.sinh(self.s*L) )
      u = np.dot(self.q[self.t1:self.t2, 2:self.nm+2],a) * c
      self.ax1[0].plot(t, u)
      self.ax1[0].set_xlabel('t (s)')
      self.ax1[0].set_ylabel('u0 (m/s)')

      #plot fourier transform of u
      #self.ax1[1].specgram(u - np.mean(u), Fs=self.fs*c/L)
      U = np.fft.rfft( u - np.mean(u) )
      f = np.fft.rfftfreq(len(t), 1./self.fs)
      self.ax1[1].plot(f * c / L, abs(U))
      self.ax1[1].set_xlabel('f (Hz)')

      fp1, fp2 = calc_fixed_points(self.params)
      jac = calc_jacobian(fp1, self.params)
      evs, evecs = np.linalg.eig(jac)
      self.ax1[2].plot(evs.real, c/L/2/pi*evs.imag, 'o')
      self.ax1[2].set_xlabel('Re(eig) (Hz)')
      self.ax1[2].set_ylabel('Im(eig) (Hz)')

      self.fig1.canvas.draw()

  def plot_page_two(self):
      terms = by_term(self.q, self.params)
      (radiation, shedding, vocal_fold_friction, dissipation, 
        elastic, linear, quadratic, cubic) = terms
      
      [ax.cla() for ax in self.ax2.flatten()]

      self.ax2[0,0].plot(radiation)
      self.ax2[0,0].set_title("radiation")
      self.ax2[1,0].plot(shedding)
      self.ax2[1,0].set_title("shedding")
      self.ax2[2,0].plot(vocal_fold_friction)
      self.ax2[2,0].set_title("vocal_fold_friction")
      self.ax2[3,0].plot(dissipation)
      self.ax2[3,0].set_title("dissipation")
      self.ax2[0,1].plot(elastic)
      self.ax2[0,1].set_title("elastic")
      self.ax2[1,1].plot(linear)
      self.ax2[1,1].set_title("linear")
      self.ax2[2,1].plot(quadratic)
      self.ax2[2,1].set_title("quadratic")
      self.ax2[3,1].plot(cubic)
      self.ax2[3,1].set_title('cubic')
      labels=['mode_' + str(n) for n in range(self.nm)]
      self.ax2[0,0].legend(self.ax2[0,0].lines, labels)

      self.fig2.canvas.draw()

  def update_plots(self):
      if self.page == 1:
        self.plot_page_one()
      if self.page == 2:
        self.plot_page_two()

  def on_T_change(self, text):
      if len(text) != 0:
        try:
          self.T = float(text)
        except:
          raise ValueError('Input must be a float')

      name = 'T'
      unit = self.controls_dict[name][4]
      conversion = self.controls_dict[name][5]
      label_text = (name + ' (' + str(round(self.T * conversion, 4)) + 
      ' ' + unit + ')')
      self.controls_dict[name][1].setText(label_text)
      
      self.t = np.arange(0, self.T, 1/self.fs)
      self.steps_label.setText('n_steps=' + str(len(self.t))) 
      self.pbar.setValue(0)
      #self.controls_dict['t1'][0].setText('0')
      #self.controls_dict['t2'][0].setText(str(len(self.t)))

  def on_fs_change(self, text):
      if len(text) != 0:
        try:
          self.fs = float(text)
        except:
          raise ValueError('Input must be a float')

      name = 'fs'
      unit = self.controls_dict[name][4]
      conversion = self.controls_dict[name][5]
      label_text = (name + ' (' + str(self.fs * conversion) + 
      ' ' + unit + ')')
      self.controls_dict[name][1].setText(label_text)
      self.pbar.setValue(0)

      self.t = np.arange(0, self.T, 1. / self.fs)
      self.steps_label.setText('n_steps=' + str(len(self.t))) 
      #self.controls_dict['t1'][0].setText('0')
      #self.controls_dict['t2'][0].setText(str(len(self.t)))

  def on_t1_change(self, text):
      if len(text) != 0:
        try:
          self.t1 = int(text)
        except:
          raise ValueError('Input must be an integer')

      name = 't1'
      unit = self.controls_dict[name][4]
      conversion = self.controls_dict[name][5]
      label_text = (name + ' (' + str(self.t1 * conversion) + 
      ' ' + unit + ')')
      self.controls_dict[name][1].setText(label_text)
      self.update_plots()

  def on_t2_change(self, text):
      if len(text) != 0:
        try:
          self.t2 = int(text)
        except:
          raise ValueError('Input must be an integer')

      name = 't2'
      unit = self.controls_dict[name][4]
      conversion = self.controls_dict[name][5]
      label_text = (name + ' (' + str(self.t2 * conversion) + 
      ' ' + unit + ')')
      self.controls_dict[name][1].setText(label_text)
      self.update_plots()

  def on_p_change(self, text):
      if len(text) != 0:
        try:
          self.p = float(text)
        except:
          raise ValueError('Input must be a float')

        name = 'p'
        unit = self.controls_dict[name][4]
        conversion = self.controls_dict[name][5]
        label_text = (name + ' (' + str(self.p * conversion) + 
        ' ' + unit + ')')
        self.controls_dict[name][1].setText(label_text)
        self.pbar.setValue(0)

  def on_r_folds_change(self, text):
      if len(text) != 0:
        try:
          self.r_folds = float(text)
        except:
          raise ValueError('Input must be a float')

        name = 'r_folds'
        unit = self.controls_dict[name][4]
        conversion = self.controls_dict[name][5]
        label_text = (name + ' (' + str(self.r_folds * conversion) + 
        ' ' + unit + ')')
        self.controls_dict[name][1].setText(label_text)
        self.pbar.setValue(0)

  def on_omega_d_change(self, text):
    if len(text) != 0:
      try:
        self.omega_d = float(text)
      except:
        raise ValueError('Input must be a float')

      name = 'omega_d'
      unit = self.controls_dict[name][4]
      conversion = self.controls_dict[name][5]
      label_text = (name + ' (' + str(self.omega_d * conversion) + 
      ' ' + unit + ')')
      self.controls_dict[name][1].setText(label_text)
      self.pbar.setValue(0)

  def on_Gamma_change(self, text):
    if len(text) != 0:
      try:
        self.Gamma = float(text)
      except:
        raise ValueError('Input must be a float')

      name = 'Gamma'
      unit = self.controls_dict[name][4]
      conversion = self.controls_dict[name][5]
      label_text = (name + ' (' + str(self.Gamma * conversion) + 
      ' ' + unit + ')')
      self.controls_dict[name][1].setText(label_text)
      self.pbar.setValue(0)


  def on_r_folds_change(self, text):
      if len(text) != 0:
        try:
          self.r_folds = float(text)
        except:
          raise ValueError('Input must be a float')

        name = 'r_folds'
        unit = self.controls_dict[name][4]
        conversion = self.controls_dict[name][5]
        label_text = (name + ' (' + str(self.r_folds * conversion) + 
        ' ' + unit + ')')
        self.controls_dict[name][1].setText(label_text)
        self.pbar.setValue(0)

  def on_r_mouth_change(self, text):
      if len(text) != 0:
        try:
          self.r_mouth = float(text)
        except:
          raise ValueError('Input must be a float') 

        name = 'r_mouth'
        unit = self.controls_dict[name][4]
        conversion = self.controls_dict[name][5]
        label_text = (name + ' (' + str(self.r_mouth * conversion) + 
        ' ' + unit + ')')
        self.controls_dict[name][1].setText(label_text)
        if self.r_mouth == 0:
          self.s = j*np.zeros(self.nm)+np.array([(2*n+1)*pi/2 for n in range(self.nm)])
        else:
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
    unit = self.controls_dict[name][4]
    conversion = self.controls_dict[name][5]
    label_text = (name + ' (' + str(self.nm * conversion) + 
    ' ' + unit + ')')
    self.controls_dict[name][1].setText(label_text)
    if self.nm != 0:
      self.q0 = np.zeros(2*self.nm+3)
      self.q0[1] = 1
      #self.q0 = fp1 + 1e-12 * abs(np.random.rand(2*self.nm+1))
      self.s = cp.calc_spatial_evs(r_mouth=self.r_mouth/L, nm=self.nm, 
      num_seeds=300, max_x=20, max_y=2)
      
    self.pbar.setValue(0)

  def on_push(self):
      self.params = cp.calc_parameters(self.p, self.r_folds, self.r_folds, 
        self.omega_d, self.Gamma, self.s.imag, self.s.real, self.nm)
      self.run_simulation()
      self.update_plots()


if __name__ == "__main__":
  app = QApplication(sys.argv)
  main = MainWindow()
  main.show()
  sys.exit(app.exec_())