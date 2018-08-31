# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import seaborn as sns
from scipy import stats

from libdalton import constants as const
from libdalton import fileio
from libdalton import simulate

class Plot:
    def __init__(self, ana):
        self.sim_type = ana.sim_type
        self.data = ana.prop
        self.out_plot = ana.out_plot
        self.percent_start = ana.percent_start
        self.percent_stop = ana.percent_stop
        self.pdict = ana.pdict
        
        self.fig_width = 7.5
        self.fig_height = 4.5
        self.fig_size = (self.fig_width, self.fig_height)
        self.img_format = 'pdf'
        
        ppi = const.POINTSPERINCH
        p_im = const.PERCENTIMAGEPLOT
        self.n_maxpoints = int(np.floor(2.0 * ppi * self.fig_width * p_im))
        self.n_terms = len(self.data.keys()) - 1
        
        self.line_colors = {}
        self.line_priorities = {}
        self.leg_labels = {}
        self.leg_priorities = {}
        
        self.ekeys = []
        
        for key in self.data:
            if 'e_' in key:
                self.line_colors[key] = self.pdict[key][2]
                self.line_priorities[key] = self.pdict[key][1]
                self.leg_labels[key] = self.pdict[key][0]
                self.leg_priorities[key] = self.pdict[key][3]
                self.ekeys.append(key)
                
        self.ekeys = sorted(list(self.ekeys), key = lambda x: self.pdict[x][1])

class TrajectoryPlot(Plot):
    def __init__(self, ana):
        super().__init__(ana)
        self.name = ana.name
        self.y_label = 'Energy Terms (kcal/mol)'
        
        if self.sim_type == 'md':
            self.title = 'Molecular Dynamics Simulation of '
            self.x_label = 'Time (ps)'
            self.x_var = 'time'
        elif self.sim_type == 'mc':
            self.title = 'Metropolis Monte-Carlo Simulation of '
            self.x_label = 'Configuration Number'
            self.x_var = 'conf'
        
        self.title += self.name
        
        self.line_linewidth = 1.0
        self.line_alpha = 1.0
        self.line_label = ''
        
        self.leg_linewidth = 4.0
        self.leg_ncol = 4
        self.leg_fontsize = 10.5
        self.leg_pos = (0.5, -0.05)
        self.leg_alpha = 1.0
        self.leg_edgecolor = '#000000'
        
        self.grid_color = '#000000'
        self.grid_alpha = 0.10
        self.grid_linestyle = '-'        
    
        self.title_fontsize = 14
        self.yaxis_fontsize = 12
        self.xaxis_fontsize = 12
    
        self.x_minpad = 0.001
        self.x_maxpad = 0.001
        self.y_minpad = 0.040
        self.y_maxpad = 0.240
        
        self.burn_in = int(np.floor(len(self.data['e_pot']) * self.percent_start / 100.0))
        self.plot_distrib = ana.plot_distrib
        self.plot_ensemble = ana.plot_ensemble
        
    def make_plot(self):
        plt.figure(figsize=self.fig_size)
        self._get_lines()
        self._get_axes()
        self._get_x_ticks()
        self._get_y_ticks()
        self._get_labels()
        self._get_grid()
        self._get_legend()
        self._output_plot()
        
        if self.plot_distrib:
            self._energy_dist_plot()
        if self.plot_ensemble:
            self._ensemble_plot()
        
    def _get_lines(self):
        self._get_point_indices()
        self._get_x_vals()
        self._get_y_vals()
        
        for key in self.ekeys:
            plt.plot(
                    self.x_vals[key],
                    self.y_vals[key],
                    linewidth=self.line_linewidth,
                    color=self.line_colors[key],
                    alpha=self.line_alpha,
                    label=self.line_label)
        
    def _get_point_indices(self):
        self.n_confs = len(self.data[self.x_var])
        self.n_start = int(np.floor(self.percent_start * self.n_confs) / 100.0)
        self.n_stop = int(np.ceil(self.percent_stop * self.n_confs) / 100.0)
        self.n_points = min(self.n_stop - self.n_start, self.n_maxpoints)
        self.percent_ranges = np.linspace(self.n_start, self.n_stop, self.n_points + 1)
        
        for percent_range in self.percent_ranges:
            percent_range = round(percent_range)
        
        self.percent_ranges = self.percent_ranges.astype(int)
    
    def _get_x_vals(self):
        self.x_vals = {}
        
        for key in self.ekeys:
            self.x_vals[key] = np.zeros(self.n_points)
            
            for i in range(self.n_points):
                start_vals = self.percent_ranges[max(0, i-1):i+1]
                stop_vals = self.percent_ranges[i:min(i+2, self.n_points+2)]
                n_start = int(np.ceil(np.average(start_vals)))
                n_stop = int(np.ceil(np.average(stop_vals)))
                val_array = self.data[key][n_start:n_stop]
                
                if i % 2 == 0:
                    idx = n_start + np.argmax(val_array)
                else:
                    idx = n_start + np.argmin(val_array)
                
                self.x_vals[key][i] = self.data[self.x_var][idx]
                
    def _get_y_vals(self):
        self.y_vals = {}
        
        for key in self.ekeys:
            self.y_vals[key] = np.zeros(self.n_points)
            
            for i in range(self.n_points):
                start_vals = self.percent_ranges[max(0, i-1):i+1]
                stop_vals = self.percent_ranges[i:min(i+2, self.n_points+2)]
                n_start = int(np.ceil(np.average(start_vals)))
                n_stop = int(np.ceil(np.average(stop_vals)))
                val_array = self.data[key][n_start:n_stop]
                
                if i % 2 == 0:
                    val = np.amax(val_array)
                else:
                    val = np.amin(val_array)
                
                self.y_vals[key][i] = val
            
    def _get_axes(self):
        self._get_axis_bounds()
        plt.axis([self.x_low, self.x_high, self.y_low, self.y_high])
    
    def _get_axis_bounds(self):
        self.x_mindat = self.data[self.x_var][0]
        self.x_maxdat = self.data[self.x_var][len(self.data[self.x_var])-1]
        self.x_rangedat = self.x_maxdat - self.x_mindat
        
        self.x_min = self.x_mindat + (self.percent_start * self.x_rangedat) / 100.0
        self.x_max = self.x_mindat + (self.percent_stop * self.x_rangedat) / 100.0
        self.x_range = self.x_max - self.x_min
        self.x_low = self.x_min - self.x_minpad * self.x_range
        self.x_high = self.x_max - self.x_maxpad * self.x_range   
        self.x_rangeplot = self.x_high - self.x_low
        
        self.y_min = float('inf')
        self.y_max = float('-inf')
        
        for key in self.ekeys:
            self.y_min = min(self.y_min, np.amin(self.data[key][self.burn_in:]))
            self.y_max = max(self.y_max, np.amax(self.data[key][self.burn_in:]))
        
        self.y_range = self.y_max - self.y_min            
        self.y_low = self.y_min - self.y_minpad * self.y_range
        self.y_high = self.y_max + self.y_maxpad * self.y_range  
        self.y_rangeplot = self.y_high - self.y_low
        
    def _get_tick_resolution(self, lower_bound, upper_bound):
        axis_range = upper_bound - lower_bound
        range_power = int(np.floor(np.log10(axis_range)))
        lead_digit = axis_range * 10**(-range_power)
        
        if lead_digit <= 2.0:
            base_digit = 2.0
        elif lead_digit <= 5.0:
            base_digit = 5.0
        else:
            base_digit = 10.0
        
        return base_digit * 10**(range_power)
    
    def _get_ticks(self, lower_bound, upper_bound, axis):
        tick_res = self._get_tick_resolution(lower_bound, upper_bound)
        tick_delta = 0.1 * tick_res
        tick_min = tick_delta * (int(lower_bound / tick_delta) - 1)
        tick_max = tick_delta * (int(upper_bound / tick_delta) + 1)
        
        n_ticks = round((tick_max - tick_min) / tick_delta) + 1
        ticks = list(np.linspace(tick_min, tick_max, n_ticks))
        
        for i in range(len(ticks)-1, -1, -1):
            if ticks[i] < lower_bound or ticks[i] > upper_bound:
                ticks.pop(i)
        
        tick_labels = [''  for i in range(len(ticks))]
        
        for i, tick in enumerate(ticks):
            exp = int(np.floor(np.log10(max(1, abs(tick)))) / 3)
            val = tick_delta * round(tick / tick_delta) / 10**(3*exp)
            
            if int(val) == val:
                val = int(val)
            
            tick_labels[i] = '%s%s' % (val, const.TICCHARS[exp])
        
        if axis == 'x':
            plt.xticks(ticks, tick_labels)
        elif axis == 'y':
            plt.yticks(ticks, tick_labels)
    
    def _get_x_ticks(self):
        self._get_ticks(self.x_low, self.x_high, 'x')
        
    def _get_y_ticks(self):
        self._get_ticks(self.y_low, self.y_high, 'y')
    
    def _get_labels(self):
        plt.title(self.title, fontsize=self.title_fontsize)
        plt.xlabel(self.x_label, fontsize=self.xaxis_fontsize)
        plt.ylabel(self.y_label, fontsize=self.yaxis_fontsize)
    
    def _get_grid(self):
        plt.grid(
                color = self.grid_color,
                alpha = self.grid_alpha,
                linestyle = self.grid_linestyle)
    
    def _get_legend(self):
        self.ekeys = sorted(list(self.ekeys), key = lambda x: self.pdict[x][3])
        
        for key in self.ekeys:
            plt.plot(0, 0,
                     linewidth = self.leg_linewidth,
                     color = self.line_colors[key],
                     alpha = self.leg_alpha,
                     label = self.leg_labels[key])
        
        self.legend = plt.legend(
                loc = 'upper center',
                ncol = self.leg_ncol,
                fontsize = self.leg_fontsize,
                bbox_to_anchor = self.leg_pos,
                framealpha = self.leg_alpha)
        
        self.legend.get_frame().set_edgecolor(self.leg_edgecolor)
    
    def _output_plot(self):
        plt.savefig(self.out_plot, format=self.img_format,
                    bbox_extra_artists=(self.legend,), bbox_inches='tight')
        plt.close()
        
    def _energy_dist_plot(self):
        fig, ax = plt.subplots(figsize=(16,10), ncols=5, nrows=2)
        fig.suptitle('Energy Terms Distribution (kcal/mol)', fontsize=24, position=(0.5,0.96))
        
        for row in range(2):
            for col in range(5):
                key = const.PROPERTYKEYS[row * 5 + col + 2]
                key_label = const.PROPERTYDICTIONARY[key]
                
                sns.distplot(self.data[key][self.burn_in:], bins=40,
                             ax=ax[row, col],
                             axlabel=key_label[0], color=key_label[2])
                
                xticks = ax[row,col].get_xticks()
                yticks = ax[row,col].get_yticks()
                xticks_spacing = xticks[1] - xticks[0]
                yticks_spacing = yticks[1] - yticks[0]
                xminor_locator = MultipleLocator(0.5 * xticks_spacing)
                yminor_locator = MultipleLocator(0.5 * yticks_spacing)

                ax[row,col].xaxis.set_minor_locator(xminor_locator)
                ax[row,col].yaxis.set_minor_locator(yminor_locator)
                ax[row,col].grid(which = 'major',
                                 color = self.grid_color,
                                 alpha = self.grid_alpha,
                                 linestyle = self.grid_linestyle)
                ax[row,col].grid(which = 'minor',
                                 color = self.grid_color,
                                 alpha = 0.5*self.grid_alpha,
                                 linestyle = '--')
                ax[row,col].set_xlabel(key_label[0], fontsize=12)
        
        out_dist = os.path.splitext(self.out_plot)[0] + '_edist.pdf'
        fig.savefig(out_dist, format=self.img_format, bbox_inches='tight', pad_inches=0.5)
    
    def _ensemble_plot(self):
        fig, ax = plt.subplots(figsize=(7.5,4.5), ncols=1, nrows=2)
        fig.suptitle('Ensemble Properties of Simulation', fontsize=24, position=(0.5,1.12))
        
        for row in range(2):
            key = const.PROPERTYKEYS[row + 12]
            key_label = const.PROPERTYDICTIONARY[key]
            ax[row].plot(self.data['time'][self.burn_in:], self.data[key][self.burn_in:],
                           linewidth=self.line_linewidth,
                           color=key_label[2],
                           alpha=self.line_alpha,
                           label=key_label[0])
            
            ax[row].set_xlim([self.data['time'][self.burn_in:].min(),
                             self.data['time'].max()])

            xticks = ax[row].get_xticks()
            yticks = ax[row].get_yticks()
            xticks_spacing = xticks[1] - xticks[0]
            yticks_spacing = yticks[1] - yticks[0]
            xminor_locator = MultipleLocator(0.5 * xticks_spacing)
            yminor_locator = MultipleLocator(0.5 * yticks_spacing)

            ax[row].xaxis.set_minor_locator(xminor_locator)
            ax[row].yaxis.set_minor_locator(yminor_locator)
            ax[row].grid(which = 'major',
                             color = self.grid_color,
                             alpha = self.grid_alpha,
                             linestyle = self.grid_linestyle)
            ax[row].grid(which = 'minor',
                             color = self.grid_color,
                             alpha = 0.5*self.grid_alpha,
                             linestyle = '--')
            ax[row].set_xlabel(key_label[0], fontsize=12)
        
        out_ensemble = os.path.splitext(self.out_plot)[0] + '_ensemble.pdf'
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        fig.savefig(out_ensemble, format=self.img_format, bbox_inches='tight', pad_inches=0.3)

class Analysis:
    def __init__(self, infile_name):
        self.infile = os.path.realpath(infile_name)
        self.indir = os.path.dirname(self.infile)
        self.pdict = const.PROPERTYDICTIONARY
        
        self.sim_type = ''
        self.sim_file = ''
        self.sim_dir = ''
        self.out_plot = 'plot.pdf'
        self.in_energy = ''
        self.in_geom = ''
        self.percent_start = 0.0
        self.percent_stop = 100.0
        
        self.plot_distrib = False
        self.plot_ensemble = False
        
        self.read_data()
    
    def read_data(self):
        fileio.get_analysis_data(self)
        self._read_files()
        self._read_properties()
    
    def run(self):
        if self.sim_type == 'mc':
            self.plot_distrib = True
        if self.sim_type == 'md':
            self.plot_ensemble = True
            
        self._get_energy_stats()
        print(fileio.get_print_averages(self))            
        self.tplt = TrajectoryPlot(self)
        self.tplt.make_plot()
        
    def _read_files(self):
        cwd = os.getcwd()
        os.chdir(self.sim_dir)
        sim = simulate.Simulation(self.sim_file)
        os.chdir(cwd)
        
        self.in_energy = sim.out_energy
        self.in_geom = sim.out_geom
        self.name = sim.mol.name
    
    def _read_properties(self):
        self.prop = fileio.get_properties(self.in_energy)
    
    def _get_energy_stats(self):
        burn_in = int(np.floor(len(self.prop['e_pot']) * self.percent_start / 100.0))
        self.eavg = {}
        self.estd = {}
        self.emin = {}
        self.emax = {}
        
        for key in self.pdict:
            if key in self.prop and key in const.PROPERTYKEYS[:-2]:
                key_data = self.prop[key]
                self.eavg[key] = np.average(key_data[burn_in:])
                self.estd[key] = np.std(key_data[burn_in:])
                self.emin[key] = np.amin(key_data[burn_in:])
                self.emax[key] = np.amax(key_data[burn_in:])