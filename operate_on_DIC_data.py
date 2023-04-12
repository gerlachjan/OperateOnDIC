# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 21:11:38 2023

@author: Gerlach
"""

import os 
import io
import imageio
import pandas as pd
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable

#==============================================================================
# GENERAL SETTINGS
#==============================================================================

plt.close('all')

#colors
iulblue = np.array([55, 96, 146])/256
c1 = np.array([127, 127, 127])/256      #grey
c2 = np.array([222, 0, 0])/256          #red
c3 = np.array([0, 176, 80])/256         #green
c4 = np.array([210, 210, 210])/256      #light grey
c5 = np.array([238, 127, 0])/256        #orange
c6 = np.array([240, 182, 0])/256        #creame yellow


#==============================================================================
# FUNCTION DEFINITIONS
#==============================================================================

def create_movie(png_dir):
    images = []
    idx_img = 1

    for file_name in sorted(os.listdir(png_dir)):
        print(['processing image_'+str(idx_img)])
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
            idx_img = idx_img + 1

    output_file = os.path.join(png_dir, 'movie.mp4')
    imageio.mimwrite(output_file, images, fps=6)


def get_delta_value_matrix(data):
    n = data.shape[0]
    dvalues = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            dvalue = np.abs(data[i] - data[j])
            dvalues[i, j] = dvalue
            dvalues[j, i] = dvalue
    return dvalues

def get_n_smallest_values_of_col(data,col,n):
    subarray = data[:,col]
    min_val = np.min(subarray)
    min_idx = np.argmin(subarray)
    
    sorted_indices = np.argsort(subarray)
    
    min_indices = sorted_indices[:n]
    min_values = subarray[min_indices] 
    
    # Convert the 1D indices to 2D indices
    i_indices = np.zeros_like(min_indices) + col
    j_indices = min_indices
    
    # Print the results
    for i in range(len(min_indices)):
        print("Distance #%d: %.10f (indices: %d, %d)" % (i+1, min_values[i], i_indices[i], j_indices[i]))
    
    return min_values, j_indices

def quadratic_regression(xdata, ydata):
    coeff = np.polyfit(xdata, ydata, 2)
    return np.poly1d(coeff)

def get_area_between_function(f1, f2, x_range):
    diff_f = f1 - f2
    area_between = quad(diff_f, x_range[0], x_range[-1])[0]
    return area_between

def iul_plot_style(fig,ax, xlim = False, ylim=False,xlabel='xlabel',ylabel='ylabel', fontsize = 16, loc = 'lower right', safe_plot=False, legend=True, noaxis=False, infobox_str = None, infobox_pos =[0.63, 0.935], count = 0, plot_name = 'plot'):
    
    ax.tick_params(width=2, length=4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, linewidth=1)
    # ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.set_xlabel(xlabel,fontsize=fontsize, fontweight='bold', family='Arial')
    ax.set_ylabel(ylabel,fontsize=fontsize,fontweight='bold', family='Arial')
   
    font = fm.FontProperties(family='Arial',size = fontsize)
    font_legend = fm.FontProperties(family='Arial',size = fontsize-5)
    plt.rcParams["mathtext.fontset"] = "cm"
    
    ax.xaxis.set_tick_params()
    ax.yaxis.set_tick_params()
    
    for label in ax.get_xticklabels():
        label.set_fontproperties(font)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font)
    
    if legend:
        ax.legend(prop=font_legend,facecolor='white', fancybox=False, framealpha=1,edgecolor='black', loc=loc)
    
    if noaxis == True:
        plt.axis('off')
    
    if infobox_str:
        props = dict(boxstyle='square', facecolor='white', alpha=1)
        # infobox_str = '\n'.join((r'$\mu=%.2f$' % (0.3, ),r'$\mathrm{E}=%d$MPa' % (210000, ),r'$\sigma_f=%.2f$MPa' % (550, )))
        ax.text(infobox_pos[0],infobox_pos[1], infobox_str, transform=ax.transAxes, fontsize=fontsize-5, family='Arial', verticalalignment='center', bbox=props)
    
    fig.set_size_inches(6.30, 3.54)
    # fig.set_size_inches(3.15, 3.54)
    
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
    if safe_plot:
        label = f'{count:03}'
        fig.savefig('plots/'+plot_name+label+'.png', dpi=300)
        # fig.savefig('plots/plot'+label+'.tiff', dpi=300)
        # fig.savefig('plots/plot'+label+'.eps', dpi=300)
        # fig.savefig('plots/plot'+label+'.svg', dpi=300)

def iul_plot_style3D(fig,ax, xlim = False, ylim=False, zlim=False,xlabel='xlabel',ylabel='ylabel',zlabel='zlabel', fontsize = 16, loc = 'lower right', safe_plot=False, legend=True, noaxis=False, infobox_str = None, infobox_pos =[0.63, 0.935], count = 0, plot_name = 'plot'):
    
    ax.tick_params(width=2, length=4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.grid(True, linewidth=1)
    # ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.set_xlabel(xlabel,fontsize=fontsize, fontweight='bold', family='Arial')
    ax.set_ylabel(ylabel,fontsize=fontsize,fontweight='bold', family='Arial')
    ax.set_zlabel(zlabel,fontsize=fontsize,fontweight='bold', family='Arial')
    
    font = fm.FontProperties(family='Arial',size = fontsize)
    font_legend = fm.FontProperties(family='Arial',size = fontsize-5) 
    plt.rcParams["mathtext.fontset"] = "cm"
    
    ax.xaxis.set_tick_params()
    ax.yaxis.set_tick_params()
    ax.zaxis.set_tick_params()
    
    for label in ax.get_xticklabels():
        label.set_fontproperties(font)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font)
    for label in ax.get_zticklabels():
        label.set_fontproperties(font)
    
    if legend:
        ax.legend(prop=font_legend,facecolor='white', fancybox=False, framealpha=1,edgecolor='black', loc=loc)
        
    if noaxis == True:
        plt.axis('off')
        
    if infobox_str:
        props = dict(boxstyle='square', facecolor='white', alpha=1)
        ax.text(infobox_pos[0],infobox_pos[1], infobox_str, transform=ax.transAxes, fontsize=fontsize-5, family='Arial', verticalalignment='center', bbox=props)
        
    fig.set_size_inches(10, 7.5)
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
    if safe_plot:
        label = f'{count:03}'
        fig.savefig('plots/'+plot_name+label+'.png', dpi=300)
        # fig.savefig('plots/plot'+label+'.tiff', dpi=300)
        # fig.savefig('plots/plot'+label+'.eps', dpi=300)
        # fig.savefig('plots/plot'+label+'.svg', dpi=300)

def get_true_cross_section_from_DIC(folder_path,filename_init,filename_last,filename_fracture,header_row=6, w0=10, t0=1.5, n_eval=31,plot_all_frames = False):
    
    # get intial sepcimen coordinates
    df_init = pd.read_csv(os.path.join(folder_path, filename_init),delimiter=';',skiprows=header_row-1)
    df_last = pd.read_csv(os.path.join(folder_path, filename_last),delimiter=';',skiprows=header_row-1)
    
    # calculate initial cross-section at notch
    A0 = w0*t0
    
    # based on the coordinate system of the DIC data, changes might have to be done!
    xdata_init = df_init['x']
    ydata_init = df_init['y']
    zdata_init = df_init['z']
    zdata_init_sym = zdata_init-t0


    # get index of minium strain --> should be at location of smallest cross-section
    thickness_strain = df_last['thickness_strain']
    min_strain = np.min(thickness_strain)
    min_strain_index = np.argmin(thickness_strain)


    # get indices of points within minimum cross-section of notched tensile test specimen
    y_coordinate = df_last['x']
    y_dvalues = get_delta_value_matrix(y_coordinate)
    y_dvalues_min, j_indices = get_n_smallest_values_of_col(y_dvalues, min_strain_index, n_eval)
    
    y_range_init = np.linspace(ydata_init[j_indices].min(),ydata_init[j_indices].max(),100)
    f_top_init = quadratic_regression(ydata_init[j_indices], zdata_init[j_indices])
    f_bottom_init = quadratic_regression(ydata_init[j_indices], zdata_init_sym[j_indices])

    area_between_init = get_area_between_function(f_top_init, f_bottom_init, y_range_init)


    cross_section = []
    count = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            if file_name == filename_fracture:
                break
            count = count + 1

            df = pd.read_csv(os.path.join(folder_path, file_name),delimiter=';',skiprows=header_row-1)
            
            xdata = df['x']
            ydata = df['y']
            
            # eliminate rigid body translation due to coordinate movement in thickness direction
            u_z = -(df['displacement_z']-min(df['displacement_z'])) 
            zdata = zdata_init+ u_z
            zdata_sym = zdata_init-t0-u_z
            
            y_range = np.linspace(ydata[j_indices].min(),ydata[j_indices].max(),100)
            f_top = quadratic_regression(ydata[j_indices], zdata[j_indices])
            f_bottom = quadratic_regression(ydata[j_indices], zdata_sym[j_indices])

            area_between = get_area_between_function(f_top, f_bottom, y_range)
            
            relative_change = (area_between-area_between_init)/area_between_init*100
            area_between_corrected = A0*area_between/area_between_init
            cross_section.append(area_between_corrected)
            
            print("Current cross-section A: %.5f mm^2"  % (area_between_corrected))
                        
            if plot_all_frames:
                fig, ax = plt.subplots()
                ax.plot(y_range_init, f_top_init(y_range_init),color=c1, zorder = 2)
                ax.plot(y_range_init, f_bottom_init(y_range_init),color=c1,zorder = 2)
                ax.scatter(ydata_init[j_indices], zdata_init[j_indices],color = 'white',zorder = 2, edgecolor=c1)
                ax.scatter(ydata_init[j_indices], zdata_init_sym[j_indices],color = 'white',zorder = 2, edgecolor=c1)
                # Plot the two functions and the area between them
                ax.fill_between(y_range_init, f_top_init(y_range_init), f_bottom_init(y_range_init), where=(y_range_init>=y_range_init[0]) & (y_range_init<=y_range_init[-1]), alpha=0.5, color=c1, label=f"Initial cross-section $A_0$ : {A0:.2f} mm$^2$",zorder = 1)
                
                
                ax.plot(y_range, f_top(y_range),color=iulblue, zorder = 2)
                ax.plot(y_range, f_bottom(y_range),color=iulblue,zorder = 2)
                ax.scatter(ydata[j_indices], zdata[j_indices],color = 'white',zorder = 2, edgecolor=iulblue)
                ax.scatter(ydata[j_indices], zdata_sym[j_indices],color = 'white',zorder = 2, edgecolor=iulblue)
                # Plot the two functions and the area between them
                ax.fill_between(y_range, f_top(y_range), f_bottom(y_range), where=(y_range>=y_range[0]) & (y_range<=y_range[-1]), alpha=0.5, color=iulblue, label=f"Cross-section $A$ : {area_between_corrected:.2f} mm$^2$",zorder = 1)
                iul_plot_style(fig, ax,xlim=[-5,5], ylim=[-2.4,0.3],xlabel='Y-coordinate in mm', ylabel= 'Z-coordinate in mm',safe_plot=True,noaxis=False, legend = True, count = count, plot_name = 'A')
                
                
                # 3D visualization of equivalent mises strain of DIC-data   
                xdata_combined = np.concatenate((xdata, xdata))
                ydata_combined = np.concatenate((ydata, ydata))
                zdata_combined = np.concatenate((zdata, zdata_sym))
                color_data_combined = np.concatenate((df['mises_strain'], df['mises_strain']))
                
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
    
                p = ax.scatter(xdata_combined, ydata_combined, zdata_combined, c=color_data_combined, cmap='jet', zorder=2)
    
                cbar = fig.colorbar(p, shrink=0.5)
                cbar.set_label(r'Equivalent strain $\varepsilon_\mathrm{eq}$', rotation=90, fontsize=16, labelpad=5, fontweight='bold', family='Arial')
                cbar.ax.tick_params(labelsize=16)
                min_val = 0.00
                max_val = 0.38
                cbar.mappable.set_clim(min_val, max_val)
                cbar.set_ticks([min_val, min_val + 1/2*abs(max_val - min_val), max_val]) 
                ax.view_init(15, 120)
                iul_plot_style3D(fig, ax,xlim=[-9.5,11.5], ylim=[-8,8],zlim=[-2,0],xlabel='X-coordinate', ylabel= 'Y-coordinate',zlabel='Z-coordinate', safe_plot=True,count = count,noaxis =True, legend=False, plot_name = 'DIC')
    
            
    # Visualization of points (last frame before fracture) used for cross-section calculation -> Adjust n_val if more or less points should be included 
    fig, ax = plt.subplots() #create single plot
    ax.scatter(ydata, xdata,color = c1,zorder = 1, edgecolor=c1, label = 'Initial specimen geometry')
    ax.scatter(ydata[j_indices], xdata[j_indices],color = 'red',zorder = 2, edgecolor=c1, label = "Points used for cross-section calculation")
    iul_plot_style(fig, ax,xlim=[-11,11], ylim=[-12,14],xlabel='Y-coordinate in mm', ylabel= 'X-coordinate in mm',safe_plot=True,noaxis=False,legend = True, plot_name = 'points_to_eval')  

    # Visualization of equivalent mises strain (last frame before fracture) 
    fig, ax = plt.subplots() #create single plot
    p1 = ax.scatter(ydata, xdata, c= df['mises_strain'], cmap='jet',zorder = 1)
    cbar = fig.colorbar(p1)
    cbar.set_label(r'Equivalent strain $\varepsilon_\mathrm{eq}$', rotation=90, fontsize=16, labelpad=5, fontweight='bold', family='Arial')
    cbar.ax.tick_params(labelsize=16)
    min_val = round(np.min(df['mises_strain']),2)
    max_val = round(np.max(df['mises_strain']),2)
    cbar.mappable.set_clim(min_val, max_val)
    cbar.set_ticks([min_val, min_val + 1/2*abs(max_val - min_val), max_val])
    iul_plot_style(fig, ax,xlim=[-11,11], ylim=[-12,14],xlabel='Y-coordinate in mm', ylabel= 'X-coordinate in mm',safe_plot=True,noaxis=False,legend = False, plot_name = 'mises2D')

    
    # Visualization of cross-section area (last frame before fracture)
    fig, ax = plt.subplots()
    ax.plot(y_range_init, f_top_init(y_range_init),color=c1, zorder = 2)
    ax.plot(y_range_init, f_bottom_init(y_range_init),color=c1,zorder = 2)
    ax.scatter(ydata_init[j_indices], zdata_init[j_indices],color = 'white',zorder = 2, edgecolor=c1)
    ax.scatter(ydata_init[j_indices], zdata_init_sym[j_indices],color = 'white',zorder = 2, edgecolor=c1)
    # Plot the two functions and the area between them
    ax.fill_between(y_range_init, f_top_init(y_range_init), f_bottom_init(y_range_init), where=(y_range_init>=y_range_init[0]) & (y_range_init<=y_range_init[-1]), alpha=0.5, color=c1, label=f"Initial cross-section $A_0$ : {A0:.2f} mm$^2$",zorder = 1)
    ax.plot(y_range, f_top(y_range),color=iulblue, zorder = 2)
    ax.plot(y_range, f_bottom(y_range),color=iulblue,zorder = 2)
    ax.scatter(ydata[j_indices], zdata[j_indices],color = 'white',zorder = 2, edgecolor=iulblue)
    ax.scatter(ydata[j_indices], zdata_sym[j_indices],color = 'white',zorder = 2, edgecolor=iulblue)
    # Plot the two functions and the area between them
    ax.fill_between(y_range, f_top(y_range), f_bottom(y_range), where=(y_range>=y_range[0]) & (y_range<=y_range[-1]), alpha=0.5, color=iulblue, label=f"Cross-section $A$ : {area_between_corrected:.2f} mm$^2$",zorder = 1)
    iul_plot_style(fig, ax,xlim=[-5,5], ylim=[-2.4,0.3],xlabel='Y-coordinate in mm', ylabel= 'Z-coordinate in mm',safe_plot=True,noaxis=False, legend = True, plot_name = 'cross_section')
    
    
    # 3D visualization of equivalent mises strain of DIC-data   
    xdata_combined = np.concatenate((xdata, xdata))
    ydata_combined = np.concatenate((ydata, ydata))
    zdata_combined = np.concatenate((zdata, zdata_sym))
    color_data_combined = np.concatenate((df['mises_strain'], df['mises_strain']))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(xdata_combined, ydata_combined, zdata_combined, c=color_data_combined, cmap='jet', zorder=2)
    cbar = fig.colorbar(p, shrink=0.5)
    cbar.set_label(r'Equivalent strain $\varepsilon_\mathrm{eq}$', rotation=90, fontsize=16, labelpad=5, fontweight='bold', family='Arial')
    cbar.ax.tick_params(labelsize=16)
    cbar.mappable.set_clim(min_val, max_val)
    cbar.set_ticks([min_val, min_val + 1/2*abs(max_val - min_val), max_val]) 
    ax.view_init(15, 120)
    iul_plot_style3D(fig, ax,xlim=[-9.5,11.5], ylim=[-8,8],zlim=[-2,0],xlabel='X-coordinate', ylabel= 'Y-coordinate',zlabel='Z-coordinate', safe_plot=True,noaxis =True, legend=False, plot_name = 'mises3D')        
    

    print('Operations on DIC-data finished!')                
    return np.array(cross_section)

#==============================================================================
# MAIN PROGRAM
#==============================================================================
if __name__ == '__main__':
 
    test_cross_section = get_true_cross_section_from_DIC('DIC_frames','Flächenkomponente 1_0.000 s.csv','Flächenkomponente 1_38.444 s.csv','Flächenkomponente 1_38.555 s.csv', plot_all_frames = False)
    # create_movie('plots')







