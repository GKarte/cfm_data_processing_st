# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:46:34 2025

@author: Gregor
"""



import numpy as np
import pandas as pd
from io import BytesIO
import os
import sys
# import personal plotting lib
sys.path.append(r"C:\Users\Gregor\Documents\GitHub\matplotlib_templates")
import plots_gk as pgk



def calc_mean_pressures(df_raw):
    
    calib_corr_df = df_raw.loc["calibration correction mbar"]
    sensor_heights_df = df_raw.loc["sensor height"] # store height data
    df_data_raspi = df_raw.drop(labels=["sensor number","sensor range","sensor height","calibration correction mbar"], axis=0) # drop non-pressure rows

    # calculate mean values of pressure values and convert to df
    p_mean = df_data_raspi.mean().to_frame() # mean values of measured an corrected (auto calib) values
    p_std = df_data_raspi.std().to_frame() # mean values of measured an corrected (auto calib) values
    p_mean_not_corr = (df_data_raspi.mean()+calib_corr_df).to_frame() # mean values of measured values (not corrected)

    # merge dfs
    df_out = pd.merge(sensor_heights_df, p_mean, left_index=True, right_index=True) # add heights
    df_out = pd.merge(df_out, p_std, left_index=True, right_index=True) # add std
    df_out = pd.merge(df_out, p_mean_not_corr, left_index=True, right_index=True) # add calib corr
    df_out.columns = ["h/m", "p_mean/mbar", "p_std/mbar", "p_mean_not_corr/mbar"]
    # sort according to name
    df_out["index"] = df_out.index
    try:
        df_out = df_out.drop(["gm_ZR", "gm_ZL"])
    except KeyError:
        pass
    df_out["sort1"] = df_out['index'].str.extract(r'([a-zA-Z]*)')
    df_out["sort2"] = df_out['index'].str.extract('(\d+)', expand=False).astype(int)
    df_out = df_out.sort_values(['sort1', 'sort2'], ascending=[True, True])
    df_out = df_out.drop(labels=["index","sort1","sort2"], axis=1)
    
    return df_out, df_data_raspi


def gm_signal_to_Vdot(time_array, signal_array):
    pulses = signal_array - np.roll(signal_array, 1) # extract signal changes (pulses)
    pulses_ind = np.where(pulses[1:] > 30)[0] + 1 # extract indexes of positive (ramp-up) signal changes
    pulse_times = time_array[pulses_ind] # get timestamps from pulses
    pulse_dts = pulse_times[1:] - pulse_times[:-1] # get time intervals between pulses
    pulse_dt_glob = sum(pulse_dts) # get time intervall between first and last pulse
    V_dots = (0.1 / pulse_dts * 3600) # V_dot measurements (m^3/h) from pulses
    V_dot_mean = V_dots.mean() # mean V_dot (m^3/h) from pulses
    V_dot_std = V_dots.std() # std from V_dots (m^3/h)
    n_V_dot = V_dots.size # number of V_dot measurements
    V_dot_glob = len(pulse_dts) * 0.1 / pulse_dt_glob * 3600 # mean V_dot (m^3/h) from first and last pulse
    # print(pulse_times)
    # print(pulse_dts)
    # print(pulse_dt_glob)
    # print(V_dots)
    # print(V_dot_mean)
    # print(V_dot_std)
    # print(n_V_dot)
    # print(V_dot_glob)
    # print()
    return V_dots, V_dot_mean, V_dot_std, n_V_dot, V_dot_glob
    


def calc_Vdots_out(df_in):
    
    
    # adjust timestamps (index of df_in) to start with 0
    time_array_str = list(df_in.index)
    time_array = np.zeros(len(time_array_str))
    for i,timestr in enumerate(time_array_str):
        time_array[i] = sum([a*b for a,b in zip([3600,60,1], map(float,timestr.split(" ")[1].split(':')))])
    df_in["t_tot"] = time_array
    time_array = time_array-time_array[0]
    df_in.index = time_array  
    df_out = df_in
    time_array = np.array(time_array)
    
    # gm ZR
    gm_zr_signal = np.array(list(df_out["gm_ZR"])) # signal
    V_dots_CR, V_dot_CR_mean, V_dot_CR_std, n_V_dot_CR, V_dot_CR_glob = gm_signal_to_Vdot(time_array, gm_zr_signal)
    
    # gm ZL
    gm_zl_signal = np.array(list(df_out["gm_ZL"])) # signal
    V_dots_GR, V_dot_GR_mean, V_dot_GR_std, n_V_dot_GR, V_dot_GR_glob = gm_signal_to_Vdot(time_array, gm_zl_signal) 
    # stats dataframe
    df_Vdot_stats = pd.DataFrame(index =['Vdot_mean / m^3/h', 'Vdot_std / m^3/h', 'n_V_dot / -', 'V_dot_glob / m^3/h'])
    df_Vdot_stats["CR"] = [V_dot_CR_mean, V_dot_CR_std, n_V_dot_CR, V_dot_CR_glob]
    df_Vdot_stats["GR"] = [V_dot_GR_mean, V_dot_GR_std, n_V_dot_GR, V_dot_GR_glob]
    # all Vdots dataframe
    df_V_dots = pd.DataFrame()
    df_V_dots["CR"] = V_dots_CR
    df_V_dots["GR"] = pd.Series(V_dots_GR)
    
    
    return df_out, df_Vdot_stats, df_V_dots










# Path to the folder containing CSV files
folder_path = r"./raspi_csvs"

# List all files in the folder
files = os.listdir(folder_path)

# Filter for .csv files only
csv_files = [f for f in files if f.endswith(".csv")]

# Read each CSV file into a DataFrame and store in a dictionary
filenames = []
dfs_raw = []
dfs_out = []

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df_raw = pd.read_csv(file_path, sep=",", header=0, index_col=0, engine='python')
    # extract filename
    filename = os.path.splitext(file)[0]
    print(f"Loaded {file} with shape {df_raw.shape}")

    # calc mean pressures and 
    df_p, df_data_raspi = calc_mean_pressures(df_raw)
    p_h = df_p[["h/m","p_mean/mbar"]]
    
    # add to lists
    filenames.append(filename)
    dfs_raw.append(df_raw)
    dfs_out.append(df_p)
    
    
    # correction of GF19->GF20
    # p_h.loc["GF20"] = p_h.loc["GF19"]
    # p_h.loc["GF20"]["h"] = p_h.loc["GF19"]["h"] + 0.05
    # p_h.drop(labels=["GF19"], axis=0, inplace=True)



    # extract data for individual parts of system
    p_h = df_p[["h/m","p_mean/mbar"]]
    p_h.columns = ["h", "p"]
    p_h = p_h.iloc[:, [1, 0]]
    
    
    p_h_gf = p_h[p_h.index.str.contains("GF")].sort_values(by=['h']).to_numpy()
    p_h_seg = p_h[p_h.index.str.contains("SEG")].sort_values(by=['h']).to_numpy()
    p_h_conv = p_h[p_h.index.str.contains("CONV")].sort_values(by=['h']).to_numpy()
    p_h_ris = p_h[p_h.index.str.contains("RIS")].sort_values(by=['h']).to_numpy()
    p_h_llsl = p_h[p_h.index.str.contains("CON_L")].to_numpy()
    p_h_llsr = p_h[p_h.index.str.contains("CON_R")].to_numpy()
    p_h_uls = p_h[p_h.index.str.contains("ULS")].to_numpy()
    p_h_ils = p_h[p_h.index.str.contains("ILS")].to_numpy()
    p_h_zr = p_h[p_h.index.str.contains("ZR")].to_numpy()
    p_h_zl = p_h[p_h.index.str.contains("ZL")].to_numpy()

    """sensor corrections"""
    # p_h_ris[-2][0]=(p_h_ris[-1][0]-p_h_ris[-3][0])/(p_h_ris[-1][1]-p_h_ris[-3][1])*(p_h_ris[-2][1]-p_h_ris[-3][1])+p_h_ris[-3][0]
    # 
    
    # connnect connections/loop seals and cyclone pressures with reactors
    p_h_llsl2 = np.concatenate(([p_h_gf[0,:]], p_h_llsl, [p_h_seg[1,:]]), axis=0)
    p_h_llsr2 = np.concatenate(([p_h_seg[1,:]], p_h_llsr, [p_h_ris[1,:]]), axis=0)
    p_h_ils2 = np.concatenate(([p_h_gf[-1,:]], p_h_ils, [p_h_gf[5,:]]), axis=0)
    p_h_uls2 = np.concatenate(([p_h_ris[-1,:]], p_h_uls, [p_h_gf[9,:]]), axis=0)
    p_h_zl2 = np.concatenate(([p_h_gf[-1,:]], [p_h_zl[-1,:]]), axis=0)
    p_h_zr2 = np.concatenate(([p_h_ris[-1,:]], [p_h_zr[-1,:]]), axis=0)
    
    # dp/dh CCC
    # p_h_gf_df = p_h[p_h.index.str.contains("GF")].sort_values(by=['h'])
    # p_h_ccc = p_h_gf_df[p_h_gf_df["h"] > 0.3].to_numpy()
    # h_mean_ccc = (p_h_ccc[1:,1] + p_h_ccc[:-1,1])/2
    # dh_ccc = p_h_ccc[1:,1] - p_h_ccc[:-1,1]
    # dp_ccc = p_h_ccc[1:,0] - p_h_ccc[:-1,0]
    # dp_dh_ccc = (dp_ccc/dh_ccc)
    
    # plot 1: complete
    fig,ax = pgk.create_plot(figsize=(4.2, 3.6), dpi=300, x_range=(-10,140), y_range=(-0.65,2), x_label="$p$ / mbar", y_label="$h$ / m", grid=False, grid_fine=False, title=False)
    ax.plot(p_h_gf[:,0], p_h_gf[:,1], marker=".", c=pgk.colors[1], label="GR", markeredgecolor='black', markeredgewidth=0.5)
    ax.plot(p_h_seg[:,0], p_h_seg[:,1], marker=".", c=pgk.colors[2], label="SG", markeredgecolor='black', markeredgewidth=0.5)
    # ax.plot(p_h_conv[:,0], p_h_conv[:,1], marker=".", c=pgk.colors[0], label="converter")
    ax.plot(p_h_ris[:,0], p_h_ris[:,1], marker=".", c=pgk.colors[3], label="CR", markeredgecolor='black', markeredgewidth=0.5)
    ax.plot(p_h_llsl2[:,0], p_h_llsl2[:,1], "--",c="darkgray", label="connections")
    ax.plot(p_h_llsr2[:,0], p_h_llsr2[:,1], "--", c="darkgray")
    ax.plot(p_h_zl2[:,0], p_h_zl2[:,1], "--", c="darkgray")
    ax.plot(p_h_zr2[:,0], p_h_zr2[:,1], "--", c="darkgray")
    ax.plot(p_h_ils2[:,0], p_h_ils2[:,1], "--", c="darkgray")
    ax.plot(p_h_uls2[:,0], p_h_uls2[:,1], "--", c="darkgray")
    ax.legend(frameon=True, edgecolor="k")


    
    
    


    # calc V_dot at cyclone outlets based on >>Gasuhr<< pulses
    # df_data_raspi, df_Vdot_stats, df_Vdots = calc_Vdots_out(df_data_raspi)



# create excel output simple (RaPi)
# output = BytesIO()
# writer = pd.ExcelWriter(output, engine = 'xlsxwriter')
# df_p.to_excel(writer, sheet_name="p_mean", float_format="%.5f", startrow=0, index=True)
# df_Vdot_stats.to_excel(writer, sheet_name="Vdot_stats", float_format="%.5f", startrow=0, index=True)
# df_Vdots.to_excel(writer, sheet_name="Vdot_raw", float_format="%.5f", startrow=0, index=True)
# df_data_raspi.to_excel(writer, sheet_name="RasPi", float_format="%.5f", startrow=0, index=True)
# writer.close()
    



    
        
        
        