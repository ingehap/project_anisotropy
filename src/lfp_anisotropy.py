#!/usr/bin/env python3

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
import numpy as np
import sys
from scipy import optimize
from misc import np2list, cut_below, cut_above, get_rms

def equinor_gamma(vcl, phit, a, b, c):
    phit_mod = np.where(phit < 0.01, 0.01, phit)
    return a * vcl**b * 10**((-1.0) * c * phit_mod)

def optimal_a(gamma_simp_opt, vec):
    return np.nansum(gamma_simp_opt * vec) / np.nansum(vec * vec)

def gamma_simp(dtm, rm, vst, rhob, lfp_c44_log):
    vmud = 0.3048e6 / dtm
    kfl = 1e-6 * rm * vmud**2
    mst = 1e-6 * rm * vst**2
    tmp = 1.0 / (1.0/mst - 1.0/kfl)
    lfp_c66_simp = np.where(tmp > 0.0, tmp, np.nan)
    vs_hor_simp = 1e3 * np.sqrt(lfp_c66_simp / rhob)
    lfp_dts_hor_simp = 0.3048e6 / vs_hor_simp
    gamma_simp_calc = (lfp_c66_simp - lfp_c44_log) / (2 * lfp_c44_log)
    return gamma_simp_calc, lfp_dts_hor_simp  

def generate_synt_gamma(vcl, phit):
    gamma_0_5_synt = equinor_gamma(vcl, phit, 0.5, 1.5, 4.5)
    gamma_0_6_synt = equinor_gamma(vcl, phit, 0.6, 1.5, 4.5)
    gamma_0_7_synt = equinor_gamma(vcl, phit, 0.7, 1.5, 4.5)
    gamma_0_8_synt = equinor_gamma(vcl, phit, 0.8, 1.5, 4.5)
    gamma_0_9_synt = equinor_gamma(vcl, phit, 0.9, 1.5, 4.5)
    gamma_1_0_synt = equinor_gamma(vcl, phit, 1.0, 1.5, 4.5)
    gamma_1_1_synt = equinor_gamma(vcl, phit, 1.1, 1.5, 4.5)
    gamma_1_2_synt = equinor_gamma(vcl, phit, 1.2, 1.5, 4.5)
    gamma_1_3_synt = equinor_gamma(vcl, phit, 1.3, 1.5, 4.5)
    gamma_1_4_synt = equinor_gamma(vcl, phit, 1.4, 1.5, 4.5)
    gamma_1_5_synt = equinor_gamma(vcl, phit, 1.5, 1.5, 4.5)
    gamma_1_6_synt = equinor_gamma(vcl, phit, 1.6, 1.5, 4.5)
    gamma_1_7_synt = equinor_gamma(vcl, phit, 1.7, 1.5, 4.5)
    gamma_1_8_synt = equinor_gamma(vcl, phit, 1.8, 1.5, 4.5)
    gamma_1_9_synt = equinor_gamma(vcl, phit, 1.9, 1.5, 4.5)
    gamma_2_0_synt = equinor_gamma(vcl, phit, 2.0, 1.5, 4.5)
    gamma_2_1_synt = equinor_gamma(vcl, phit, 2.1, 1.5, 4.5)
    gamma_2_2_synt = equinor_gamma(vcl, phit, 2.2, 1.5, 4.5)
    gamma_2_3_synt = equinor_gamma(vcl, phit, 2.3, 1.5, 4.5)
    gamma_2_4_synt = equinor_gamma(vcl, phit, 2.4, 1.5, 4.5)
    gamma_2_5_synt = equinor_gamma(vcl, phit, 2.5, 1.5, 4.5)
    return  gamma_0_5_synt, gamma_0_6_synt, gamma_0_7_synt, gamma_0_8_synt, \
            gamma_0_9_synt, gamma_1_0_synt, gamma_1_1_synt, gamma_1_2_synt, \
            gamma_1_3_synt, gamma_1_4_synt, gamma_1_5_synt, gamma_1_6_synt, \
            gamma_1_7_synt, gamma_1_8_synt, gamma_1_9_synt, gamma_2_0_synt, \
            gamma_2_1_synt, gamma_2_2_synt, gamma_2_3_synt, gamma_2_4_synt, \
            gamma_2_5_synt

def optimal_mud_prop(dt_min, dt_max, dt_steps, rho_min, rho_max, rho_steps, rhob, vs, vcl, vcl_cut, vst, lfp_c44_log, phit):
    rms_mud = np.inf
    best_dt_mud = np.inf
    best_rho_mud = np.inf
    lfp_dts_log = 304800.0 / vs
    dtmud = np2list(dt_min, dt_max, dt_steps)
    rhomud = np2list(rho_min, rho_max, rho_steps)
    for dtm in dtmud:
        for rm in rhomud:
            lfp_gamma_simp, lfp_dts_hor_simp = gamma_simp(dtm, rm, vst, rhob, lfp_c44_log)          
            dts_restr = cut_below(vcl, lfp_dts_log, vcl_cut)
            hor_simp_restr = cut_below(vcl, lfp_dts_hor_simp, vcl_cut)
            diff = get_rms(dts_restr - hor_simp_restr)
            if diff < rms_mud:
                rms_mud = diff
                best_dt_mud = dtm
                best_rho_mud = rm
    gamma_simp_opt, _ = gamma_simp(best_dt_mud, best_rho_mud, vst, rhob, lfp_c44_log)
    return best_dt_mud, best_rho_mud, rms_mud, gamma_simp_opt

def second_order_parameters(gamma_simp_opt, vcl, vcl_cut, phit, a_min, a_max, b_min, b_max):
    # Objective function
    def second_order(x):
        return get_rms(equinor_gamma(vcl, phit, x[0], x[1], 4.5) - gamma_simp_opt)
    # General estimation procedure
    def estimate_values(chosen_method):
        opt_res_obj = optimize.minimize(second_order, np.array([1.2, 1.5]), method = chosen_method, bounds = ((a_min, a_max), (b_min, b_max)))
        result = opt_res_obj.fun
        a2 = opt_res_obj.x[0]
        b2 = opt_res_obj.x[1]
        rms = get_rms(result - equinor_gamma(vcl, phit, a2, b2, 4.5))
        return a2, b2, rms
    a2_lbfgsb, b2_lbfgsb, rms_lbfgsb = estimate_values('L-BFGS-B')
    a2_p, b2_p, rms_p = estimate_values('Powell')
    a2_tnc, b2_tnc, rms_tnc = estimate_values('TNC')
    return a2_lbfgsb, b2_lbfgsb, rms_lbfgsb, a2_p, b2_p, rms_p, a2_tnc, b2_tnc, rms_tnc

def third_order_parameters(gamma_simp_opt, vcl, vcl_cut, phit, a_min, a_max, b_min, b_max, c_min, c_max):
    # Objective function
    def third_order(x):
        return get_rms(equinor_gamma(vcl, phit, x[0], x[1], x[2]) - gamma_simp_opt)
    # General estimation procedure
    def estimate_values(chosen_method):
        opt_res_obj = optimize.minimize(third_order, np.array([1.2, 1.5, 4.5]), method = chosen_method, bounds = ((a_min, a_max), (b_min, b_max), (c_min, c_max)))
        result = opt_res_obj.fun
        a3 = opt_res_obj.x[0]
        b3 = opt_res_obj.x[1]
        c3 = opt_res_obj.x[2]
        rms_a3b3 = get_rms(result - equinor_gamma(vcl, phit, a3, b3, c3))
        return a3, b3, c3, rms_a3b3
    a3_lbfgsb, b3_lbfgsb, c3_lbfgsb, rms_lbfgsb = estimate_values('L-BFGS-B')
    a3_p, b3_p, c3_p, rms_p = estimate_values('Powell')
    a3_tnc, b3_tnc, c3_tnc, rms_tnc = estimate_values('TNC')    
    return a3_lbfgsb, b3_lbfgsb, c3_lbfgsb, rms_lbfgsb, a3_p, b3_p, c3_p, rms_p, a3_tnc, b3_tnc, c3_tnc, rms_tnc

def write_up(rhob, vcl, vcl_cut, vs, vst, dt_min, dt_max, dt_steps, rho_min, rho_max, rho_steps, phit, a_min, a_max, b_min, b_max, c_min, c_max):
    lfp_c44_log = 1e-6 * rhob * vs**2
    # Mud properties
    best_dt_mud, best_rho_mud, rms_mud, gamma_simp_opt = optimal_mud_prop(dt_min, dt_max, dt_steps, rho_min, rho_max, rho_steps, rhob, vs, vcl, vcl_cut, vst, lfp_c44_log, phit)
    # First order parameter calculation
    a1 = optimal_a(gamma_simp_opt, equinor_gamma(vcl, phit, 1.0, 1.5, 4.5))
    rms_a1 = get_rms(equinor_gamma(vcl, phit, a1, 1.5, 4.5) - gamma_simp_opt)
    # Second order parameter calculations
    a2_lbfgsb, b2_lbfgsb, rms2_lbfgsb, \
    a2_p, b2_p, rms2_p, \
    a2_tnc, b2_tnc, rms2_tnc = second_order_parameters(gamma_simp_opt, vcl, vcl_cut, phit, a_min, a_max, b_min, b_max)
    # Third order parameter calculations
    a3_lbfgsb, b3_lbfgsb, c3_lbfgsb, rms3_lbfgsb, \
    a3_p, b3_p, c3_p, rms3_p, \
    a3_tnc, b3_tnc, c3_tnc, rms3_tnc = third_order_parameters(gamma_simp_opt, vcl, vcl_cut, phit, a_min, a_max, b_min, b_max, c_min, c_max)
    print("\n********************************************")
    print("**** Start - Anisotropy summary report ****\n\n")
    print("1. Explanation of the estimation of the mud properties in the Equinor anisotropy model\n")
    print("Mud properties in Equinor anisotropy model is here obtained by minimalization of")
    print("RMS(lfp_dts_log - lfp_dts_hor_simp) constrained by the following conditions")
    print("     A. lfp_vcldry < vcl_dry_cut")
    print("     B. 304800  / DTST < vmud\n")
    print("2. Optimal mud properties\n")
    print("{:<15} {:<29}".format("     Best dt_mud                ", str(best_dt_mud) + " us/ft"))
    print("{:<15} {:<29}".format("     Best dt_mud                ", str(best_rho_mud) + " g/cm3"))
    print("{:<15} {:<29}".format("     Minimal RMS for mud  ", str(rms_mud) + " us/ft"))
    print("\n3. One-parameter solution (a)\n")
    print("     Best a parameter = ", a1)
    print("     Minimal RMS for a minimalization = ", rms_a1, "\n")
    print("4. Two-parameter solution (a, b)\n")
    print("     Best a parameter:")
    print("        L-BFGS-B =                ", a2_lbfgsb)
    print("        Powell =                    ", a2_p)
    print("        TNC =                        ", a2_tnc)
    print("     Best b parameter:")
    print("        L-BFGS-B =                ", b2_lbfgsb)
    print("        Powell =                    ", b2_p)
    print("        TNC =                        ", b2_tnc)
    print("     Minimal RMS for a and b minimalization:")
    print("        L-BFGS-B =                ", rms2_lbfgsb)
    print("        Powell =                    ", rms2_p)
    print("        TNC =                        ", rms2_tnc)
    print("\n5. Three-parameter solution (a, b, c)\n")
    print("     Best a parameter:")
    print("        L-BFGS-B =                ", a3_lbfgsb)
    print("        Powell =                    ", a3_p)
    print("        TNC =                        ", a3_tnc)
    print("     Best b parameter:")
    print("        L-BFGS-B =                ", b3_lbfgsb)
    print("        Powell =                    ", b3_p)
    print("        TNC =                        ", b3_tnc)
    print("     Best c parameter:")
    print("        L-BFGS-B =                ", c3_lbfgsb)
    print("        Powell =                    ", c3_p)
    print("        TNC =                        ", c3_tnc)
    print("     Minimal RMS for for a, b and c minimalization:")
    print("        L-BFGS-B =                ", rms3_lbfgsb)
    print("        Powell =                    ", rms3_p)
    print("        TNC =                        ", rms3_tnc)
    print("\n6. Output logs\n")
    print("Output logs are written to your chosen output set\n\n")
    print("**** End - Anisotropy summary report ****")
    print("*******************************************\n")
    return a2_lbfgsb, b2_lbfgsb, a2_p, b2_p, a2_tnc, b2_tnc, a3_lbfgsb, b3_lbfgsb, c3_lbfgsb, a3_p, b3_p, c3_p, a3_tnc, b3_tnc, c3_tnc

