from pathlib import Path
import os
import re

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import ConvexHull, QhullError

from coypus.datacontainer import DataContainer as DC
from debyetools.aux_functions import load_cell
from debyetools.pairanalysis import pair_analysis as pa

cmap = plt.colormaps['rainbow']


# ---------------------------
# Small utilities
# ---------------------------
def pad_array(arr, target_length, fill_value=None):
    arr = np.asarray(arr, dtype=object)
    if len(arr) >= target_length:
        return arr
    padding = np.full(target_length - len(arr), fill_value, dtype=object)
    return np.concatenate((arr, padding))


def float_or_none(value):
    try:
        return float(value)
    except Exception:
        return None


def compute_filtered_convex_hull(points):
    points = np.asarray(points)

    if len(points) == 0:
        return points

    if len(points) < 3:
        return points[points[:, 1] <= 0]

    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
    except QhullError:
        hull_points = points

    return hull_points[hull_points[:, 1] <= 0]


def new_temperature_table(T):
    ntemps = len(T)
    table = DC()
    for _ in range(ntemps - 1):
        table.new_row()
    table.add_attribute("T")
    setattr(table, "T", pad_array(T, ntemps + 1))
    return table


def add_padded_attribute(table, name, values, ntemps):
    table.add_attribute(name)
    setattr(table, name, pad_array(values, ntemps + 1))


def get_variant_map(base_dir):
    s_list = sorted(os.listdir(base_dir))
    return {s: sorted(os.listdir(base_dir / s)) for s in s_list}


def iter_variants(variant_map):
    for s, sx_list in variant_map.items():
        for sx in sx_list:
            yield s, sx


def get_variant_strings(variant_map):
    return [f"{s}/{sx}" for s, sx in iter_variants(variant_map)]


def extract_temperature_axis(tprops_all_dict):
    first_s = next(iter(tprops_all_dict))
    first_sx = next(iter(tprops_all_dict[first_s]))
    return np.array(tprops_all_dict[first_s][first_sx]["T"])

def extract_xy_per_temperature(table, NATOMS):
    valid_temperatures = [Ti for Ti in table.T if Ti is not None]

    xy_per_temp = {
        f"{float(Ti):.5f}": {"x": [], "y": [], "label": []}
        for Ti in valid_temperatures
    }

    for row in table:
        if not row.T or row.T[0] is None:
            continue

        temp_key = f"{float(row.T[0]):.5f}"

        for key in row.keys()[2:]:
            value = row[key][0]
            if value is None:
                continue

            xy_per_temp[temp_key]["x"].append((NATOMS - int(key.split("_")[2])) / NATOMS)
            xy_per_temp[temp_key]["y"].append(float(value))
            xy_per_temp[temp_key]["label"].append("_".join(key.split("_")[-2:]))

    return xy_per_temp

def keep_only_volume_columns(table):
    keys_to_remove = [
        key for key in table.keys()
        if "V_s" not in key and key not in {"ix", "T"}
    ]
    table.remove_keys(keys_to_remove)
    return table


def parse_formula(formula):
    types_list = re.findall(r"[A-Z][a-z]*", formula)
    total = len(types_list)
    c_fe = types_list.count("Fe") / total
    c_cr = types_list.count("Cr") / total
    n_fe = types_list.count("Fe")
    return types_list, c_fe, c_cr, n_fe


# ---------------------------
# Data loading / table building
# ---------------------------
def load_tprops_by_variant(tprops, variant_map):
    tprops_all_dict = {}
    for s, sx_list in variant_map.items():
        tprops_all_dict[s] = {}
        for sx in sx_list:
            tprops_all_dict[s][sx] = tprops.filter_by_value("sname", f"{s}/{sx}").to_dict()
    return tprops_all_dict


def build_tgs_table(tprops_all_dict, outpath):
    T = extract_temperature_axis(tprops_all_dict)
    ntemps = len(T)
    table_tgs = new_temperature_table(T)

    for s, sx_dict in tprops_all_dict.items():
        for sx, values in sx_dict.items():
            T_i = np.array(values["T"])
            V_i = np.array(values["V"])
            G_i = np.array(values["G"])
            S_i = np.array(values["S"])
            H_i = G_i + T_i * S_i

            add_padded_attribute(table_tgs, f"G_{s}_{sx}", G_i, ntemps)
            add_padded_attribute(table_tgs, f"S_{s}_{sx}", S_i, ntemps)
            add_padded_attribute(table_tgs, f"H_{s}_{sx}", H_i, ntemps)
            add_padded_attribute(table_tgs, f"V_{s}_{sx}", V_i, ntemps)

    table_tgs.save(str(outpath))
    return table_tgs, T


def get_table_column_as_float_array(table, key, drop_none=True):
    values = [float_or_none(v) for v in table[key]]

    if drop_none:
        values = [v for v in values if v is not None]
        return np.array(values, dtype=float)

    return np.array([np.nan if v is None else v for v in values], dtype=float)

def build_mix_tables(
    tprops_all_dict,
    table_tgs,
    T,
    pure_fe_id,
    pure_cr_id,
    natoms,
    hmix_outpath,
    vmix_outpath,
    hform_outpath,
        Href_fe,
        Href_cr,
):
    ntemps = len(T)
    table_hmix = new_temperature_table(T)
    table_vmix = new_temperature_table(T)
    table_hform = new_temperature_table(T)

    H_cr = get_table_column_as_float_array(table_tgs, f"H_{pure_cr_id}")
    H_fe = get_table_column_as_float_array(table_tgs, f"H_{pure_fe_id}")
    V_cr = get_table_column_as_float_array(table_tgs, f"V_{pure_cr_id}")
    V_fe = get_table_column_as_float_array(table_tgs, f"V_{pure_fe_id}")

    for s, sx_dict in tprops_all_dict.items():
        n_fe = int(s.split("_")[1])
        n_cr = natoms - n_fe

        for sx in sx_dict:
            variant_id = f"{s}_{sx}"

            H_fecr = get_table_column_as_float_array(table_tgs, f"H_{variant_id}")
            V_fecr = get_table_column_as_float_array(table_tgs, f"V_{variant_id}")

            dHmix = (H_fecr - n_cr / natoms * H_cr - n_fe / natoms * H_fe)
            dVmix = (V_fecr - n_cr / natoms * V_cr - n_fe / natoms * V_fe)
            dHform = H_fecr - n_cr / natoms * Href_cr - n_fe / natoms * Href_fe

            add_padded_attribute(table_hmix, f"DHmix_{variant_id}", dHmix, ntemps)
            add_padded_attribute(table_vmix, f"DVmix_{variant_id}", dVmix, ntemps)
            add_padded_attribute(table_hform, f"DHform_{variant_id}", dHform, ntemps)

    table_hmix.save(str(hmix_outpath))
    table_vmix.save(str(vmix_outpath))
    table_hform.save(str(hform_outpath))
    return table_hmix, table_vmix, table_hform

# ---------------------------
# Convex hull / volume plots
# ---------------------------
def plot_HV_mix(ax, table_Hmix, totnats, typehv):
    # print(table_Hmix.T)

    XY_per_comp_T = {f'{float(Ti):.5f}':{'x':[], 'y':[], 'label': []} for Ti in table_Hmix.T}

    nats_al = [int(kii) for kii in set([ki.split('_')[2] for ki in table_Hmix.keys()[2:]])]
    nats_al.sort()

    Ts = [f'{Ti:.5f}' for Ti in table_Hmix.T]

    for TDi in table_Hmix:
        for k in TDi.keys()[2:]:
            XY_per_comp_T[f'{TDi.T[0]:.5f}']['x'].append((totnats-int(k.split('_')[2]))/totnats)
            XY_per_comp_T[f'{TDi.T[0]:.5f}']['y'].append(float(TDi[k][0]))
            XY_per_comp_T[f'{TDi.T[0]:.5f}']['label'].append('_'.join(k.split('_')[-2:]))

    norm = mcolors.Normalize(vmin=min(table_Hmix.T), vmax=max(table_Hmix.T))

    for Ti, v in XY_per_comp_T.items():

        # print(XY_per_comp_T[Ti])
        x = v['x']
        y = np.array(v['y'])
        y =  y/1e-5 if typehv == 'V' else y/1000
        sx = v['label']
        color = cmap(norm(float(Ti)))
        ax.plot(x, y, 'o', label=f'T={float(Ti):.0f}', color = color, alpha=0.3)


    for li, xi, yi in zip(sx, x, y):
        ax.annotate(li, xy=(xi, yi), textcoords="offset points", xytext=(15, 0), ha='center', fontsize=6)

    ax.set_xlim(0, 1)
    ax.set_xlabel(r'$x_{Cr}$ (molar)')
    if typehv == 'H':
        ax.set_ylabel(r'$\Delta H_{mix}$ (kJ/mol-atom)')
        ax.axhline(0, color='black', lw=0.5, ls='--')
    if typehv == 'V':
        ax.set_ylabel(r"$\Delta V_{mix}$ ($10e^{-5}m^3$/mol-atom)")
        ax.set_ylim(-0.1, 0.1)
        ax.axhline(0, color='black', lw=0.5, ls='--')
    if typehv == 'Hf':
        ax.set_ylabel(r'$\Delta H_{f}$ (kJ/mol-atom)')

    ax.legend(fontsize=6)

    def plot_vmix_convex_hulls_and_vegard(axxx, table_vmix, table_tgs, NATOMS, limitsVm=(-0.02, 0.02),
                                          limitsV=(0.6, 0.8)):
        xy_vmix = extract_xy_per_temperature(table_vmix, NATOMS)
        volume_table = keep_only_volume_columns(table_tgs)
        xy_volume = extract_xy_per_temperature(volume_table, NATOMS)

        ay, amix = axxx[0], axxx[1]

        norm = mcolors.Normalize(vmin=min(table_vmix.T), vmax=max(table_vmix.T))

        i = 0
        for Ti, data in xy_vmix.items():

            color = cmap(norm(float(Ti)))
            xm, ym = data["x"], np.array(data["y"]) / 1e-5
            mkr = ["$" + l.split('_')[1] + "$" for l in data['label']]

            # ax.plot(xm, ym, "o", label=f"T={float(Ti):.0f}", color=color, markerfacecolor="none")
            for xi, yi, mi in zip(xm, ym, mkr):
                amix.scatter(
                    xi, yi,
                    marker=mi,
                    edgecolors='none',
                    facecolors=color,
                    # s=25,
                )
            amix.plot([], [], color=color, marker='o', markerfacecolor='none', linestyle='none',
                      label=f"T={float(Ti):.0f}")

            if i % 3 == 0:
                xs = np.array(xy_volume[Ti]["x"])
                ys = np.array(xy_volume[Ti]["y"]) / 1e-5

                indices = np.argsort(xs)
                xs = xs[indices]
                ys = ys[indices]
                min_v = ys[0]
                max_v = ys[-1]
                # ay.plot( xs , ys,
                #     marker = "o",
                #     label=f"T={float(Ti):.0f}",
                #     color=color,
                #     markersize = 5,
                #     linestyle = "none",
                #     markerfacecolor="none",
                # )
                for xi, yi, mi in zip(xs, ys, mkr):
                    ay.scatter(
                        xi, yi,
                        marker=mi,
                        edgecolors='none',
                        facecolors=color,
                        # s=25,
                    )
                ay.plot([], [], color=color, marker='o', markerfacecolor='none', linestyle='none',
                        label=f"T={float(Ti):.0f}")
                vegard = lambda x: ((x) * max_v + (1 - x) * min_v)

                ay.plot(
                    xs,
                    [vegard(xi) for xi in xs],
                    "k--",
                    linewidth=0.5,
                )
            i += 1
        # for mi in mkr:
        #         ay.scatter(-1,-1, marker=mi, edgecolors='none', facecolors='black', label = f'sx_{mi}')

        sx = data['label']
        x, y = data['x'], np.array(data['y']) / 1e-5
        for li, xi, yi in zip(sx, x, y):
            amix.annotate(li, xy=(xi, yi), textcoords="offset points", xytext=(15, 0), ha='center', fontsize=6)

        amix.set_ylim(limitsVm[0], limitsVm[1])
        ay.set_ylim(limitsV[0], limitsV[1])
        amix.set_xlim(0, 1)
        ay.set_xlim(0, 1)

    amix.set_title(f"T = {float(Ti):.2f} K")
    amix.set_xlabel(r"$x_{Zr}$ (molar)")
    ay.set_xlabel(r"$x_{Zr}$ (molar)")
    amix.set_ylabel(r"$\Delta V_{mix}$ ($10e^{-5}m^3$/mol-atom)")
    ay.set_ylabel(r"$V$ ($10e^{-5}m^3$/mol-atom)")
    amix.axhline(0, color="black", lw=0.5, ls="--")
    amix.legend(fontsize=6, ncols=2)
    ay.legend(fontsize=6, ncols=1)


# ---------------------------
# Pair-analysis helpers
# ---------------------------
def load_variant_cell(relax_dir, variant_str):
    return load_cell(relax_dir / variant_str / "relaxation" / "CONTCAR")


def get_pair_arrays(npairs, combtypes):
    return {
        "Fe-Fe": npairs[:, combtypes.index("Fe-Fe")] if "Fe-Fe" in combtypes else [0],
        "Cr-Fe": npairs[:, combtypes.index("Cr-Fe")] if "Cr-Fe" in combtypes else [0],
        "Cr-Cr": npairs[:, combtypes.index("Cr-Cr")] if "Cr-Cr" in combtypes else [0],
    }


def get_pair_totals(npairs, combtypes, n_fe, NATOMS):
    pair_arrays = get_pair_arrays(npairs, combtypes)

    if n_fe == 0:
        fe_fe_pairs = 0
        fe_cr_pairs = 0
        cr_cr_pairs = sum(pair_arrays["Cr-Cr"])
    elif n_fe == NATOMS:
        fe_fe_pairs = sum(pair_arrays["Fe-Fe"])
        fe_cr_pairs = 0
        cr_cr_pairs = 0
    else:
        fe_fe_pairs = sum(pair_arrays["Fe-Fe"])
        fe_cr_pairs = sum(pair_arrays["Cr-Fe"])
        cr_cr_pairs = sum(pair_arrays["Cr-Cr"])

    return fe_fe_pairs, fe_cr_pairs, cr_cr_pairs


# ---------------------------
# SRO / pair plots
# ---------------------------
def find_npairs_ix(npairs, np_sum):
    for i in range(1,len(npairs)):
        sumn =sum(np.sum(npairs[0:i], axis=1))
        if np.sqrt((sumn - np_sum)**2)<0.5:
            return i

def plot_avd_sro(ax_av_iterdistance, variant_strings, RELAX_DIR, cutoff, n_pairs_ideal, lims_y=None):
    avd_Al =0
    avd_W =0
    avd_list_x = []
    avd_list_y = []

    for variant_str in variant_strings:

        types_str, cell, basis = load_variant_cell(RELAX_DIR, variant_str)
        types_list =  re.findall(r'[A-Z][a-z]*', types_str)

        c_Fe = types_list.count('Fe')/len(types_list)
        c_Cr = types_list.count('Cr')/len(types_list)

        distances,  npairs, combtypes= pa(types_str, cutoff, basis, cell)
        number_of_NN = find_npairs_ix(npairs, n_pairs_ideal)
        distances,  npairs = distances[:number_of_NN],  npairs[:number_of_NN]

        # Sum along rows to get total number of pairs per distance
        N = npairs.sum(axis=1)

        # Calculate average using dot product
        average_distance = np.dot(distances, N) / N.sum()
        avd_list_x.append(c_Cr)
        avd_list_y.append(average_distance)

        if c_Fe == 0:
            avd_W = average_distance
        if c_Fe == 1:
            avd_Al = average_distance

    xWideal = np.linspace(0, 1, 50)

    av_d_alw = [(1-x)*avd_Al + x*avd_W for x in xWideal]
    ax_av_iterdistance.plot(avd_list_x, avd_list_y, 'ko', label=f'{variant_str}_average_distance', alpha=0.5, markerfacecolor='none')

    ax_av_iterdistance.plot(xWideal, av_d_alw, 'k-', alpha=0.5, label='ideal')
    if lims_y != None:
        ax_av_iterdistance.set_ylim(lims_y[0], lims_y[1])

def plot_x_sro(axSRO3, variant_strings, RELAX_DIR, cutoff, n_pairs_ideal):
    avd_Al =0
    avd_W =0
    avd_list_x = []
    avd_list_y = []

    for variant_str in variant_strings:

        types_str, cell, basis = load_variant_cell(RELAX_DIR, variant_str)
        types_list =  re.findall(r'[A-Z][a-z]*', types_str)

        c_Fe = types_list.count('Fe')/len(types_list)
        c_Cr = types_list.count('Cr')/len(types_list)

        distances,  npairs, combtypes= pa(types_str, cutoff, basis, cell)
        number_of_NN = find_npairs_ix(npairs, n_pairs_ideal)
        distances,  npairs = distances[:number_of_NN],  npairs[:number_of_NN]

        npairs_Fe_Fe = npairs[:, combtypes.index("Fe-Fe")] if "Fe-Fe" in combtypes else np.zeros(number_of_NN)
        npairs_Cr_Cr = npairs[:, combtypes.index("Cr-Cr")] if "Cr-Cr" in combtypes else np.zeros(number_of_NN)
        if "Cr-Fe" in combtypes:
            npairs_Fe_Cr = npairs[:, combtypes.index("Cr-Fe")]
        elif "Fe-Cr" in combtypes:
            npairs_Fe_Cr = npairs[:, combtypes.index("Fe-Cr")]
        else:
            npairs_Fe_Cr = np.zeros(number_of_NN)

        N_Fe_Fe = sum(npairs_Fe_Fe)
        N_Fe_Cr = sum(npairs_Fe_Cr)
        N_Cr_Cr = sum(npairs_Cr_Cr)

        x_Fe = N_Fe_Fe/(N_Fe_Fe + N_Fe_Cr + N_Cr_Cr)
        x_Cr = N_Cr_Cr/(N_Fe_Fe + N_Fe_Cr + N_Cr_Cr)
        x_CrFe = N_Fe_Cr/(N_Fe_Fe + N_Fe_Cr + N_Cr_Cr)

        axSRO3[0][0].plot([c_Cr], [x_Fe], 'ko', label=f'{variant_str}_a_AlAl', alpha=0.5, markerfacecolor='none')
        axSRO3[1][0].plot([c_Cr], [x_CrFe], 'ko', label=f'{variant_str}_a_WAl', alpha=0.5, markerfacecolor='none')
        axSRO3[0][1].plot([c_Cr], [x_CrFe], 'ko', label=f'{variant_str}_a_AlW', alpha=0.5, markerfacecolor='none')
        axSRO3[1][1].plot([c_Cr], [x_Cr], 'ko', label=f'{variant_str}_a_WW', alpha=0.5, markerfacecolor='none')


    xWideal = np.linspace(0, 1, 50)
    xAAideal = [(1-x)**2 for x in xWideal]
    xWWideal = [(x)**2 for x in xWideal]
    xAWiedal = [2*x*(1-x) for x in xWideal]
    axSRO3[0][0].plot(xWideal, xAAideal, 'r--', alpha=0.5)
    axSRO3[0][1].plot(xWideal, xAWiedal, 'r--', alpha=0.5)
    axSRO3[1][0].plot(xWideal, xAWiedal, 'r--', alpha=0.5)
    axSRO3[1][1].plot(xWideal, xWWideal, 'r--', alpha=0.5)


def plot_alpha_sro(axSRO2, variant_strings, RELAX_DIR, cutoff, n_pairs_ideal, lims_y=None):

    for variant_str in variant_strings:

        types_str, cell, basis = load_variant_cell(RELAX_DIR, variant_str)
        types_list =  re.findall(r'[A-Z][a-z]*', types_str)

        c_Fe = types_list.count('Fe')/len(types_list)
        c_Cr = types_list.count('Cr')/len(types_list)

        distances,  npairs, combtypes= pa(types_str, cutoff, basis, cell)
        number_of_NN = find_npairs_ix(npairs, n_pairs_ideal)
        distances,  npairs = distances[:number_of_NN],  npairs[:number_of_NN]

        npairs_Fe_Fe = npairs[:, combtypes.index("Fe-Fe")] if "Fe-Fe" in combtypes else np.zeros(number_of_NN)
        npairs_Cr_Cr = npairs[:, combtypes.index("Cr-Cr")] if "Cr-Cr" in combtypes else np.zeros(number_of_NN)
        if "Cr-Fe" in combtypes:
            npairs_Fe_Cr = npairs[:, combtypes.index("Cr-Fe")]
        elif "Fe-Cr" in combtypes:
            npairs_Fe_Cr = npairs[:, combtypes.index("Fe-Cr")]
        else:
            npairs_Fe_Cr = np.zeros(number_of_NN)


        N_Fe_Fe = sum(npairs_Fe_Fe)
        N_Fe_Cr = sum(npairs_Fe_Cr)
        N_Cr_Cr = sum(npairs_Cr_Cr)

        P_FeFe = N_Fe_Fe/(N_Fe_Fe+N_Fe_Cr+N_Cr_Cr)
        P_FeCr = N_Fe_Cr/(N_Fe_Fe+N_Fe_Cr+N_Cr_Cr)
        P_CrCr = N_Cr_Cr/(N_Fe_Fe+N_Fe_Cr+N_Cr_Cr)

        alpha_FeFe = 1 - (P_FeFe / (c_Fe ** 2)) if c_Fe != 0 else float('nan')
        alpha_FeCr = 1 - (P_FeCr / (2 * c_Fe * c_Cr)) if (c_Fe != 0 and c_Cr != 0) else float('nan')
        alpha_CrCr = 1 - (P_CrCr / (c_Cr ** 2)) if c_Cr != 0 else float('nan')

        axSRO2[0][0].plot([c_Cr], [alpha_FeFe], 'ko', label=f'{variant_str}_a_AlAl', alpha=0.5, markerfacecolor='none')
        axSRO2[1][0].plot([c_Cr], [alpha_FeCr], 'ko', label=f'{variant_str}_a_WAl', alpha=0.5, markerfacecolor='none')
        axSRO2[0][1].plot([c_Cr], [alpha_FeCr], 'ko', label=f'{variant_str}_a_AlW', alpha=0.5, markerfacecolor='none')
        axSRO2[1][1].plot([c_Cr], [alpha_CrCr], 'ko', label=f'{variant_str}_a_WW', alpha=0.5, markerfacecolor='none')


    x_idwc = np.linspace(.01, .99, 100)
    axSRO2[0][1].plot(x_idwc, [1-2*min([ci,(1-ci)])/(2*ci*(1-ci)) for ci in x_idwc] , '--', label='ordered', alpha=0.5)
    axSRO2[0][1].plot([0, 1], [0, 0] , 'r--', label='random', alpha=0.5)
    axSRO2[1][0].plot([0, 1], [0, 0] , 'r--', label='random', alpha=0.5)
    axSRO2[1][0].plot(x_idwc, [1-2*min([ci,(1-ci)])/(2*ci*(1-ci)) for ci in x_idwc] , '--', label='ordered', alpha=0.5)
    axSRO2[0][0].plot([0, 1], [0, 0] , 'r--', label='random', alpha=0.5)
    axSRO2[1][1].plot([0, 1], [0, 0] , 'r--', label='random', alpha=0.5)
    axSRO2[0][0].plot([0, 1], [1, 1] , 'b--', label='ordered', alpha=0.5)
    axSRO2[1][1].plot([0, 1], [1, 1] , 'b--', label='ordered', alpha=0.5)

    if lims_y != None:
        axSRO2.set_ylim(lims_y[0], lims_y[1])

def plot_a_sro(axSRO, variant_strings, RELAX_DIR, cutoff, n_pairs_ideal, lims_y=None):

    for variant_str in variant_strings:

        types_str, cell, basis = load_variant_cell(RELAX_DIR, variant_str)
        types_list =  re.findall(r'[A-Z][a-z]*', types_str)

        c_Fe = types_list.count('Fe')/len(types_list)
        c_Cr = types_list.count('Cr')/len(types_list)

        distances,  npairs, combtypes= pa(types_str, cutoff, basis, cell)
        number_of_NN = find_npairs_ix(npairs, n_pairs_ideal)
        distances,  npairs = distances[:number_of_NN],  npairs[:number_of_NN]

        npairs_Fe_Fe = npairs[:, combtypes.index("Fe-Fe")] if "Fe-Fe" in combtypes else np.zeros(number_of_NN)
        npairs_Cr_Cr = npairs[:, combtypes.index("Cr-Cr")] if "Cr-Cr" in combtypes else np.zeros(number_of_NN)
        if "Cr-Fe" in combtypes:
            npairs_Fe_Cr = npairs[:, combtypes.index("Cr-Fe")]
        elif "Fe-Cr" in combtypes:
            npairs_Fe_Cr = npairs[:, combtypes.index("Fe-Cr")]
        else:
            npairs_Fe_Cr = np.zeros(number_of_NN)


        N_Fe_Fe = sum(npairs_Fe_Fe)
        N_Fe_Cr = sum(npairs_Fe_Cr)
        N_Cr_Cr = sum(npairs_Cr_Cr)

        N_Fe = 2*N_Fe_Fe + N_Fe_Cr
        N_Cr = 2*N_Cr_Cr + N_Fe_Cr

        #probabilities
        p_FeFe = 2*N_Fe_Fe / N_Fe if N_Fe > 0 else np.nan
        p_FeCr = N_Fe_Cr / N_Fe if N_Fe > 0 else np.nan
        p_CrCr = 2*N_Cr_Cr / N_Cr if N_Cr > 0 else np.nan
        p_CrFe = N_Fe_Cr / N_Cr if N_Cr > 0 else np.nan

        # Compute SRO Parameters
        a_FeFe = 1 - p_FeFe/c_Fe if (N_Fe > 0 and c_Fe > 0) else np.nan
        a_FeCr = 1 - p_FeCr/c_Cr if (N_Fe > 0 and c_Cr > 0) else np.nan
        a_CrCr = 1 - p_CrCr/c_Cr if (N_Cr > 0 and c_Cr > 0) else np.nan
        a_CrFe = 1 - p_CrFe/c_Fe if (N_Cr > 0 and c_Fe > 0) else np.nan

        axSRO[0][0].plot([c_Cr], [a_FeFe], 'ko', label=f'{variant_str}_a_FeFe', alpha=0.5, markerfacecolor='none')
        axSRO[1][0].plot([c_Cr], [a_CrFe], 'ko', label=f'{variant_str}_a_CrFe', alpha=0.5, markerfacecolor='none')
        axSRO[0][1].plot([c_Cr], [a_FeCr], 'ko', label=f'{variant_str}_a_FeCr', alpha=0.5, markerfacecolor='none')
        axSRO[1][1].plot([c_Cr], [a_CrCr], 'ko', label=f'{variant_str}_a_CrCr', alpha=0.5, markerfacecolor='none')

    axSRO[0][0].set_xlabel(r'$x_{Cr}$')
    axSRO[0][0].set_ylabel(r'$a_{FeFe}$')
    axSRO[0][0].set_title(r'$Fe-Fe$')

    axSRO[1][0].set_xlabel(r'$x_{Cr}$')
    axSRO[1][0].set_ylabel(r'$a_{CrFe}$')
    axSRO[1][0].set_title(r'$Cr-Fe$')

    axSRO[0][1].set_xlabel(r'$x_{Cr}$')
    axSRO[0][1].set_ylabel(r'$a_{FeCr}$')
    axSRO[0][1].set_title(r'$Fe-Cr$')

    axSRO[1][1].set_xlabel(r'$x_{Cr}$')
    axSRO[1][1].set_ylabel(r'$a_{CrCr}$')
    axSRO[1][1].set_title(r'$Cr-Cr$')

    # horizontal line at 0
    for ax in axSRO.flatten():
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    axSRO[0][0].axhline(y=1, color='k', linestyle='--', linewidth=0.5)
    axSRO[1][1].axhline(y=1, color='k', linestyle='--', linewidth=0.5)
    axSRO[1][0].axhline(y=-1, color='k', linestyle='--', linewidth=0.5)
    axSRO[0][1].axhline(y=-1, color='k', linestyle='--', linewidth=0.5)
