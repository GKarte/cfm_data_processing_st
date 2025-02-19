# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:46:34 2025

@author: Gregor
"""



import numpy as np
import pandas as pd
from io import BytesIO



def calc_mean_pressures(csv_file):
    df_raw = pd.read_csv(csv_file, sep=",", header=0, index_col=0, engine='python') # read csv
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
    df_out = df_out.drop(["gm_ZR", "gm_ZL"])
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
    
    time_array = np.array(time_array)
    
    # gm ZR
    gm_zr_signal = np.array(list(df_in["gm_ZR"])) # signal
    V_dots_CR, V_dot_CR_mean, V_dot_CR_std, n_V_dot_CR, V_dot_CR_glob = gm_signal_to_Vdot(time_array, gm_zr_signal)
    
    # gm ZL
    gm_zl_signal = np.array(list(df_in["gm_ZL"])) # signal
    V_dots_GR, V_dot_GR_mean, V_dot_GR_std, n_V_dot_GR, V_dot_GR_glob = gm_signal_to_Vdot(time_array, gm_zl_signal) 
    
    
    return df_in

def extract_gasAnalyser_section(df_GM_raw, t_start_tot, t_end_tot):
    time_array_str = list(df_GM_raw["t"])
    time_array = np.zeros(len(time_array_str))
    for i,timestr in enumerate(time_array_str):
        time_array[i] = sum([a*b for a,b in zip([3600,60,1], map(float,timestr.split(':')))])
    # time_array = time_array-time_array[0]
    df_GM_raw["t_tot"] = time_array
    df_GM = df_GM_raw[df_GM_raw["t_tot"] > t_start_tot]
    df_GM = df_GM[df_GM["t_tot"] < t_end_tot]
    
    return df_GM


def calc_gasAnalyser_stats(df_GM):
    
    CO2_mean = df_GM["CO2"].mean()
    CO2_std = df_GM["CO2"].std()
    CO2_min = df_GM["CO2"].min()
    CO2_max = df_GM["CO2"].max()
    
    return CO2_mean, CO2_std, CO2_min, CO2_max






if __name__ == "__main__":

    csv_file_raspi = "CFM_data_2025-02-17_17-20.csv"

    
    txt_file_gasMeas_CR = "CR_20250217.txt"
    txt_file_gasMeas_GR = "GR_20250217.txt"


    # calc mean pressures and 
    df_p, df_data_raspi = calc_mean_pressures(csv_file_raspi)
    
    # calc V_dot at cyclone outlets based on >>Gasuhr<< pulses
    df_Vdot = calc_Vdots_out(df_data_raspi)
    
    # print(df_Vdot)
    
    # define start- and end-time of measurement
    t_start_tot = df_Vdot.iloc[0]["t_tot"]
    t_end_tot = df_Vdot.iloc[-1]["t_tot"]

    # manual time-stamping
    # t_start_str = "16:29:13.578"
    # t_end_str = "16:32:15.238" 
    # t_start_tot = sum([a*b for a,b in zip([3600,60,1], map(float,t_start_str.split(':')))])
    # t_end_tot = sum([a*b for a,b in zip([3600,60,1], map(float,t_end_str.split(':')))])
    
    # read in gas analyser data from CR
    df_GM_CR_raw = pd.read_csv(txt_file_gasMeas_CR, sep="\t", header=0, index_col=None, engine='python') # read txt
    df_GM_CR_raw = df_GM_CR_raw[["Time", "Ch1:Conce:Vol%"]]
    df_GM_CR_raw.columns = ["t", "CO2"]
    df_GM_CR_raw["CO2"] = df_GM_CR_raw["CO2"]/100
    # read in gas analyser data from GR
    df_GM_GR_raw = pd.read_csv(txt_file_gasMeas_GR, sep="\t", header=0, index_col=None, engine='python') # read txt
    df_GM_GR_raw = df_GM_GR_raw[["Time", "Ch2:Conce:ppm"]]
    df_GM_GR_raw.columns = ["t", "CO2"]
    df_GM_GR_raw["CO2"] = df_GM_GR_raw["CO2"]/(10**6)
    
    # extract time window of data series
    df_GM_CR = extract_gasAnalyser_section(df_GM_CR_raw, t_start_tot, t_end_tot)
    df_GM_GR = extract_gasAnalyser_section(df_GM_GR_raw, t_start_tot, t_end_tot)
    print(df_GM_CR)
    print(df_GM_GR)
    
    # gas analyser stats
    GM_CR_stats = calc_gasAnalyser_stats(df_GM_CR)
    print(GM_CR_stats)
    GM_GR_stats = calc_gasAnalyser_stats(df_GM_GR)
    print(GM_GR_stats)
    
    
    
    # output = BytesIO()
    # writer = pd.ExcelWriter(output, engine = 'xlsxwriter')
    # df_out.to_excel(writer, sheet_name="data", float_format="%.5f", startrow=0, index=True)
    # writer.close()
    
    # download = st.download_button(
    #     label="Export result",
    #     data=output.getvalue(),
    #     file_name= f'mean_{csv_file.name}.xlsx'
    #     )
        
        
        
    # st.dataframe(df_out)
        
        
        