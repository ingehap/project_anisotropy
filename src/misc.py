#!/usr/bin/env python3

from datetime import datetime
from math import e, pow, sqrt, exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split

##
## Functions defined in this module, misc.py:
##
## create_plot(wellname, dataframe, curves_to_plot, depth_curve, log_curves=[])
## cut_above(select_vec, value_vec, cut)
## cut_below(select_vec, value_vec, cut)
## cut_between(select_vec, value_vec, cut_low, cut_high)
## different_values(loga, logb)
## dts_ansatz_row(nphi, rt, dt)
## gas2002cubic(t, p, g)
## get_library_contents(use_long_version = False)
## in_half_open_interval(x, a, b)
## introduction()
## mud_properties(dtst, dts, rhob, alpha)
## RHG_Analysis(a1, a2, a3, b1, b2, b3, c1, dtf, lfp_dt_log, lfp_rhob_log, lfp_rw, lfp_rt)
## sgg_g_consistency(fp_grad, low, high, inc, t, p, fname)
## slope_vshdn(lfp_nphi, lfp_rhob, lfp_rhomanc, lfp_rhog, lfp_rhoo, lfp_rhow, lfp_oil, lfp_gas)
## update_path(my_path = '/private/inp/projects')
## write_results(df, log_name_list, exp_list, metric_list, reg_type_list)
## write_rms(txt, vec)   


def create_plot(wellname, dataframe, curves_to_plot, depth_curve, log_curves=[]):
    num_tracks = len(curves_to_plot)
    fig, ax = plt.subplots(nrows=1, ncols=num_tracks, figsize=(num_tracks*2, 10))
    fig.suptitle(wellname, fontsize=20, y=1.05)
    for i, curve in enumerate(curves_to_plot):
        ax[i].plot(dataframe[curve], depth_curve)
        ax[i].set_title(curve, fontsize=14, fontweight='bold')
        ax[i].grid(which='major', color='lightgrey', linestyle='-')
        ax[i].set_ylim(depth_curve.max(), depth_curve.min())
        if i == 0:
            ax[i].set_ylabel('DEPTH (m)', fontsize=18, fontweight='bold')
        else:
            plt.setp(ax[i].get_yticklabels(), visible = False)
        if curve in log_curves:
            ax[i].set_xscale('log')
            ax[i].grid(which='minor', color='lightgrey', linestyle='-')
    plt.tight_layout()
    plt.show()


def cut_above(select_vec, value_vec, cut):
    return np.where(select_vec > cut, value_vec, np.nan)


def cut_below(select_vec, value_vec, cut):
    return np.where(select_vec < cut, value_vec, np.nan)


def cut_between(select_vec, value_vec, cut_low, cut_high):
    if (cut_low < cut_high):
        return cut_below(select_vec, cut_above(select_vec, value_vec, cut_low), cut_high)
    else:
        return np.full_like(vec, np.nan)


def different_values(loga, logb):
    if (loga == logb):
        return 0
    else:
        return 1


def dts_ansatz_row(nphi, rt, dt):
    # The training and testing of the first 9 cases are done on a set of 
    # wells in the data file "ml_data_Draupne_Heather_Fm_Selected_inp.csv" consisting
    # of wells in block 15 (NO 15/3-7 to NO 15/12-21)
    # Formation choice
    #    d = Draupne
    #    h = Heather
    #    dh = Draupne and Heather
    # Metric choice
    #    mse = mean square error
    #    mae = mean average error
    #    mdae = median average error
    #    emerge = Tronds formula

    # Tested:
    # Test \projects\w_loglan_testing\inp_misc_dts_ansatz_row
    def ansatz(nphi, rt, dt, a, b, c, d, e):
        if (dt <= 0) or (rt <= 0) or (nphi <= 0):
            return math.nan
        else:
            return a * math.pow(nphi, b) * math.exp(d * math.pow(dt, e)) / math.pow(rt, c)
            
    dts_d_mse = ansatz(nphi, rt, dt, 105.5, 0.0508, 0.0244, 3.89e-4, 1.611)
    dts_d_mae = ansatz(nphi, rt, dt, 111.1, 0.0541, 0.0246, 2.15e-4, 1.723)
    dts_d_mdae = ansatz(nphi, rt, dt, 119.4, 0.0594, 0.0249, 8.35e-5, 1.90)
    dts_h_mse = ansatz(nphi, rt, dt, 115.5, -0.00406, 0.0288, 9.61e-6, 2.369)
    dts_h_mae = ansatz(nphi, rt, dt, 111.2, -0.00450, 0.0282, 2.72e-5, 2.157)
    dts_h_mdae = ansatz(nphi, rt, dt, 120.2, -0.00342, 0.0297, 2.59e-6, 2.639)
    dts_dh_mse = ansatz(nphi, rt, dt, 103.9, 0.0431, 0.0294, 5.41e-4, 1.546)
    dts_dh_mae = ansatz(nphi, rt, dt, 108.0, 0.0436, 0.0296, 3.17e-4, 1.648)
    dts_dh_mdae = ansatz(nphi, rt, dt, 93.9, 0.0423, 0.0288, 0.00160, 1.340)
    dts_emerge = ansatz(nphi, rt, dt, 26.949, 0.039091, 0.029522, 0.20197, 0.5)
    return dts_d_mse, dts_d_mae, dts_d_mdae, dts_h_mse, dts_h_mae, dts_h_mdae, dts_dh_mse, dts_dh_mae, dts_dh_mdae, dts_emerge   


def gas2002cubic_row(t, p, g):
    """
    Input:
    t = LFP_TEMP (degC)
    p = LFP_PRESS (MPa)
    g = LFP_SGG_G (Sm3/Sm3)
    
    Output:
    LFP_RHOG (g/cm3)
    """
    r1 = 0.00831441     # Gas constant, J/g-mole deg
    tabs = t + 273.15   # absolute temperature
    mw = 28.8 * g       # molecular weight of air
    
    da0, da1, da2, da3 = -0.067, 0.0167, -0.141, 0.803
    va0, va1, va2, va3 = -0.328, 0.0302, -0.827, -0.505
    db0, db1, db2, db3 = 0.0167, 0.000851, 0.505, 0.745
    vb0, vb1, vb2, vb3 = 0.0175, 0.000911, 0.213, 0.111
    
    da = da0 + da1 * mw + (da2 * mw * tabs) * 1e-4 + (da3 * mw**2) * 1e-4
    va = va0 + va1 * mw + (va2 * mw * tabs) * 1e-6 + (va3 * mw**2) * 1e-3
    db = db0 + db1 * mw + (db2 * mw * tabs) * 1e-6 + (db3 * mw**2) * 1e-7
    vb = vb0 + vb1 * mw + (vb2 * mw * tabs) * 1e-6 + (vb3 * mw**2) * 1e-6
    
    p1 = (-1.0) * (r1 * tabs + p * db) / p
    q = da / p
    r = -da * db / p
    alpha = (3.0 * q - p1**2) / 3.0
    beta = (2.0 * pow(p1,3) - 9.0 * q * p1 + 27.0 * r)/27.0
    aa = pow(-0.5 * beta + sqrt(0.25 * pow(beta, 2) + pow(alpha, 3) / 27.0), 1.0 / 3.0)
    xhelp = (-0.5 * beta - sqrt(0.25 * pow(beta, 2) + pow(alpha, 3)/27.0))
    bb1 = pow(abs(xhelp), 1.0/3.0)
    
    if (xhelp * 1e10 < 0.0):
        bb = (-1.0) * bb1
    else:
        bb = bb1
    
    vm = aa + bb - p1 / 3.0
    rho = 1000.0 * mw / (1000.0 * vm)                                     # kg/m3
    k = 0.001 * ((vm * r1 * tabs)/ pow((vm-vb), 2) - 2 * va / pow(vm, 2)) # GPa
    v = 1000.0 * sqrt(1000.0 * k / rho)                                   # m/s
    
    return 0.001 * rho, k, v # units g/cm3, GPa, and m/s


def get_current_project_name():
    import os
    return os.environ.get("PG_PROJECT")


def get_geolog_pygg_contents(use_long_version = False):
    # Test \projects\w_loglan_testing\inp_misc_get_geolog_pygg_contents
    for lib in ["geolog", "pygg"]:
        print(f"\nContents of {lib} library:\n")
        for element in dir(lib):
            print(f"Library component {lib}.{element}")
            if use_long_version:
                print(help(lib + "." + element), "\n")


def get_number_of_none_nans(data):
    return np.count_nonzero(~np.isnan(data))


def get_rms(vec):
    return np.sqrt(np.nanmean(np.square(vec)))


def in_half_open_interval(x, a, b):
    if ((a <= x) & (x < b)):
        return True
    else:
        return False


def introduction():
    import os
    import sys
    import site
    print("\nCurrent Python version:    ", sys.version)
    print("Current working directory: ", os.getcwd())
    print("Current project name:      ", get_current_project_name())
    print("Current file name:         ", os.path.basename(__file__))
    print("Location of packages:      ", site.getsitepackages()[0], "\n")


def limit(x, a, b):
    if (x < a):
        return a
    elif (x > b):
        return b
    else:
        return x


def missing_data_flag(inputlog):
    # Test \projects\w_loglan_testing\inp_misc_missing_data_flag
    flaglog = np.ones(inputlog.shape)
    flaglog[np.isnan(inputlog)] = 0.0
    return flaglog


def mse_mlist(df, features, target, reg_type):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    if (reg_type == 'ols'):
        regr = linear_model.LinearRegression()
    else:
        regr = linear_model.Ridge(alpha=0.5)
    model = regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred, squared=False)
    mdae = median_absolute_error(y_test, y_pred)
    mlist = model.coef_.tolist()
    mlist.append(float(model.intercept_))
    return mae, mse, mdae, mlist


def mud_properties(dtst, dts, rhob, alpha):
    dtst2 = np.multiply(dtst, dtst)
    dts2rhob = np.divide(np.multiply(dts, dts), rhob)
    dtm2, dtm2_l, dtm2_u, rhom, rhom_l, rhom_u, _, _, _ = linear_regression(dts2rhob, dtst2, alpha)
    dtm = math.sqrt(dtm2)
    dtm_l = math.sqrt(dtm2_l)
    dtm_u = math.sqrt(dtm2_u)
    return dtm, dtm_l, dtm_u, rhom, rhom_l, rhom_u


def np2list(vec_low, vec_high, vec_dim):
    return list(np.linspace(vec_low, vec_high, vec_dim))


def optimal_exponent(df, log_name_list, exp_list, metric, reg_type):
    best_mlist_sofar = []
    best_metric_sofar = 1.0e100
    best_exp_sofar = 1.0e100
    lst_mae = []
    lst_mse = []
    lst_mdae = []
    df['L0'] = np.log(df[log_name_list[0]])
    df['L1'] = np.log(df[log_name_list[1]])
    df['L2'] = np.log(df[log_name_list[2]])
    target = ['L0']
    base_features = ['L1', 'L2']
    for exp in exp_list:
        new_feature = 'L3' + str(exp)
        df[new_feature] = np.power(df[log_name_list[3]], exp)
        features = base_features.copy()
        features.append(new_feature)
        mae, mse, mdae, mlist = mse_mlist(df, features, target, reg_type)
        df.drop(new_feature, axis=1, inplace=True)
        if (metric == 'mse'):
            lst_mse.append(mse)
            if (mse < best_metric_sofar):
                best_metric_sofar = mse
                best_mlist_sofar = mlist
                best_exp_sofar = exp
        elif (metric == 'mae'):
            lst_mae.append(mae)
            if (mae < best_metric_sofar):
                best_metric_sofar = mae
                best_mlist_sofar = mlist
                best_exp_sofar = exp  
        else:
            lst_mdae.append(mdae)
            if (mdae < best_metric_sofar):
                best_metric_sofar = mdae
                best_mlist_sofar = mlist
                best_exp_sofar = exp  
    df.drop('L0', axis=1, inplace=True)
    df.drop('L1', axis=1, inplace=True)
    df.drop('L2', axis=1, inplace=True)
    return best_metric_sofar, best_mlist_sofar, best_exp_sofar, lst_mae, lst_mse, lst_mdae
    
    
def predict_pickle_model_vector_input(pickle_file, input_array):
    # Should be used with gettable() / puttable()
    # Usually input_array is vp_b, pickle_file = 'RIDGE.pkl'
    import pickle
    get_model = pickle.load(open(pickle_file, 'rb'))
    return get_model.predict(input_array.reshape(-1,1))


def rhg_analysis_row(a1, a2, a3, b1, b2, b3, c1, dtf, lfp_dt_log, lfp_rhob_log, lfp_rw, lfp_rt):
    # Test \projects\w_loglan_testing\inp_misc_rhg_analysis_row
    phi = math.sqrt(lfp_rw / lfp_rt)
    BAD_DATA = (lfp_rt <= 0.0) or (lfp_rw <= 0.0) or (lfp_rhob_log <= 0.0) or (lfp_dt_log <= 0.0)
    if BAD_DATA:
        return math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
    if (phi <= 0.37):
        vp1a = a1 + b1 * math.pow(lfp_rt, -0.5) + c1 * math.pow(lfp_rt, -1.0)
        dt1a = 304800 / vp1a
        vp1b = math.pow(a2 + b2 * math.pow(lfp_rt, -0.5), 1.9) * math.pow(lfp_rhob_log, -0.5)
        dt1b = 304800 / vp1b
        dt2 = math.nan
        dt3a = math.nan
        dt3b = math.nan
    elif (phi >= 0.47):
        vp1a = math.nan
        dt1a = math.nan
        vp1b = math.nan
        dt1b = math.nan
        dt2 = (a3 + b3 * math.pow(lfp_rt, -0.5)) * math.sqrt(lfp_rhob_log)
        dt3a = math.nan
        dt3b = math.nan
    else:
        vp1a = a1 + b1 * math.pow(lfp_rt, -0.5) + c1 * math.pow(lfp_rt, -1)
        dt1a = 304800 / vp1a
        vp1b = math.pow(a2 + b2 * math.pow(lfp_rt, -0.5), 1.9) * math.pow(lfp_rhob_log, -0.5)
        dt1b = 304800 / vp1b
        dt2 = (a3 + b3 / sqrt(lfp_rt)) * sqrt(lfp_rhob_log)
        dt3a = 4.7 * dt1a - 3.7 * dt2 + 10 * (dt2 - dt1a) * math.sqrt(lfp_rw / lfp_rt)
        dt3b = 4.7 * dt1a - 3.7 * dtf + 10 * (dtf - dt1a) * math.sqrt(lfp_rw / lfp_rt)
    return vp1a, dt1a, vp1b, dt1b, dt2, dt3a, dt3b


def save_figure(exp_list, lst_err, metric, reg_type):
    x = np.array(exp_list)
    y = np.array(lst_err)
    now = str(datetime.now())
    now.replace(" ", "_")
    filename = metric + '_' + reg_type + '_' + now + '.png'
    plt.scatter(x, y)
    plt.show()
    plt.savefig(filename)
    plt.close()


def sgg_g_consistency_row(fp_grad, low, high, inc, t, p):
    # Test \projects\w_loglan_testing\inp_misc_write_sgg_g_table
    v1 = fp_grad / 0.0981
    g = low
    least_diff_so_far = 1e10
    best_g_so_far = -1
    while (g <= high):
        v2, _, _ = gas2002cubic_row(t, p, g)
        diff = abs(v2 - v1)
        if (diff < least_diff_so_far):
            least_diff_so_far = diff
            best_g_so_far = g
        g += inc
    return best_g_so_far, least_diff_so_far


def slope_vshdn(lfp_nphi, lfp_rhob, lfp_rhomanc, lfp_rhog, lfp_rhoo, lfp_rhow, lfp_oil, lfp_gas):
    # Test \projects\w_loglan_testing\inp_misc_slope_vshdn
    if (lfp_oil > 0.5):
        rho_fluid = lfp_rhoo
    elif (lfp_gas > 0.5):
        rho_fluid = lfp_rhog
    else:
        rho_fluid = lfp_rhow
    if in_half_open_interval(lfp_rhomanc, 2.60, 2.65):
        alpha = 0.040
    elif in_half_open_interval(lfp_rhomanc, 2.65, 2.67):
        alpha = 0.035
    elif in_half_open_interval(lfp_rhomanc, 2.67, 2.69):
        alpha = 0.010
    elif in_half_open_interval(lfp_rhomanc, 2.69, 2.71):
        alpha = 0.005
    else:
        alpha = 0.0
    phi_d = (lfp_rhomanc - lfp_rhob) / (lfp_rhomanc - rho_fluid)
    return phi_d / (lfp_nphi + alpha)


def update_path(my_path = '/private/inp/projects'):
    import sys
    sys.path.insert(1, my_path)


def write_results(df, log_name_list, exp_list, metric_list, reg_type_list):
    for metric in metric_list:
        for reg_type in reg_type_list:
            best_metric, best_mlist, best_exp, lst_mae, lst_mse, lst_mdae = optimal_exponent(df, log_name_list, exp_list, metric, reg_type)
            print('Regression type = ', reg_type)
            print('Metric type = ', metric)
            print('Least metric value = ', best_metric)
            print('Estimated best A = ', pow(e, best_mlist[1]))
            print('Estimated best B = ', best_mlist[0][0])
            print('Estimated best C = ', (-1.0) * best_mlist[0][1])
            print('Estimated best D = ', best_mlist[0][2])
            print('Estimated best E = ', best_exp, "\n")
            if metric == 'mse':
                save_figure(exp_list, lst_mse, 'mse', reg_type)
            elif metric == 'mae':
                save_figure(exp_list, lst_mae, 'mae', reg_type)
            else:
                save_figure(exp_list, lst_mdae, 'mdae', reg_type)


def write_rms(txt, vec):
    # Test \projects\w_loglan_testing\inp_misc_euclidean_metric
    result = np.sqrt(np.nanmean(np.square(vec)))
    print(txt, result)

