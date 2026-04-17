from pathlib import Path
import os
import re

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull, QhullError

from coypus.datacontainer import DataContainer as DC
from debyetools.aux_functions import load_cell
from debyetools.pairanalysis import pair_analysis as pa


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
):
    ntemps = len(T)
    table_hmix = new_temperature_table(T)
    table_vmix = new_temperature_table(T)

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

            dHmix = (H_fecr - n_cr / natoms * H_cr - n_fe / natoms * H_fe) / 2
            dVmix = (V_fecr - n_cr / natoms * V_cr - n_fe / natoms * V_fe) / 2

            add_padded_attribute(table_hmix, f"DHmix_{variant_id}", dHmix, ntemps)
            add_padded_attribute(table_vmix, f"DVmix_{variant_id}", dVmix, ntemps)

    table_hmix.save(str(hmix_outpath))
    table_vmix.save(str(vmix_outpath))
    return table_hmix, table_vmix

# ---------------------------
# Convex hull / volume plots
# ---------------------------
def plot_hmix_convex_hulls(table_hmix, NATOMS):
    xy_per_temp = extract_xy_per_temperature(table_hmix, NATOMS)

    fig_hull, ax_hull = plt.subplots()
    ax_hull.set_title("Convex hulls")
    ax_hull.set_xlabel(r"$x_{Cr}$ (molar)")
    ax_hull.set_ylabel(r"$\Delta H_{mix}$ (J/mol-atom)")
    ax_hull.set_xlim(0, 1)
    ax_hull.axhline(y=0, color="k", linestyle="--", linewidth=0.5)

    for Ti, data in xy_per_temp.items():
        fig, ax = plt.subplots()

        points = np.column_stack([data["x"], data["y"]])
        chull = compute_filtered_convex_hull(points)

        ax.plot(data["x"], data["y"], "o", label=f"T={Ti}", markerfacecolor="none")

        for label, x, y in zip(data["label"], data["x"], data["y"]):
            ax.annotate(label, xy=(x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6)

        if len(chull) > 0:
            chull = chull[np.argsort(chull[:, 0])]
            ax_hull.plot(chull[:, 0], chull[:, 1], label=f"T = {float(Ti):.0f}")

        ax.set_xlim(0, 1)
        ax.set_title(f"T = {float(Ti):.2f} K")
        ax.set_xlabel(r"$x_{Cr}$ (molar)")
        ax.set_ylabel(r"$\Delta H_{mix}$ (J/mol-atom)")
        ax.axhline(0, color="black", lw=0.5, ls="--")

    ax_hull.legend(fontsize=6)


def plot_vmix_convex_hulls_and_vegard(table_vmix, table_tgs, NATOMS):
    xy_vmix = extract_xy_per_temperature(table_vmix, NATOMS)
    volume_table = keep_only_volume_columns(table_tgs)
    xy_volume = extract_xy_per_temperature(volume_table, NATOMS)

    fig_hull, ax_hull = plt.subplots()
    ax_hull.set_title("Convex hulls")
    ax_hull.set_xlabel(r"$x_{Cr}$ (molar)")
    ax_hull.set_ylabel(r"$\Delta V_{mix}$ ($10e^{-5}m^3$/mol-atom)")
    ax_hull.set_xlim(0, 1)
    ax_hull.axhline(y=0, color="k", linestyle="--", linewidth=0.5)

    for Ti, data in xy_vmix.items():
        fig, axes = plt.subplots(1, 2, figsize=(15, 4))
        ay, ax = axes[0], axes[1]

        points = np.column_stack([data["x"], data["y"]])
        chull = compute_filtered_convex_hull(points)

        ax.plot(data["x"], np.array(data["y"]) / 1e-5, "o", label=f"T={Ti}", markerfacecolor="none")
        ay.plot(
            xy_volume[Ti]["x"],
            np.array(xy_volume[Ti]["y"]) / 1e-5,
            "o",
            label=f"T={Ti}",
            markerfacecolor="none",
        )

        vegard = lambda x: ((x) * xy_volume[Ti]["y"][0] + (1 - x) * xy_volume[Ti]["y"][-1]) / 1e-5
        ay.plot(
            data["x"],
            [vegard(xi) for xi in data["x"]],
            "k--",
            linewidth=0.5,
        )

        if len(chull) > 0:
            chull = chull[np.argsort(chull[:, 0])]
            ax_hull.plot(chull[:, 0], chull[:, 1] / 1e-5, label=f"T = {float(Ti):.0f}")

        ax.set_ylim(-0.08, 0.005)
        ax.set_xlim(0, 1)
        ay.set_xlim(0, 1)

        ax.set_title(f"T = {float(Ti):.2f} K")
        ax.set_xlabel(r"$x_{Cr}$ (molar)")
        ay.set_xlabel(r"$x_{Cr}$ (molar)")
        ax.set_ylabel(r"$\Delta V_{mix}$ ($10e^{-5}m^3$/mol-atom)")
        ay.set_ylabel(r"$V$ ($10e^{-5}m^3$/mol-atom)")
        ax.axhline(0, color="black", lw=0.5, ls="--")

    ax_hull.legend(fontsize=6)


# ---------------------------
# Pair-analysis helpers
# ---------------------------
def load_variant_cell(relax_dir, variant_str):
    return load_cell(relax_dir / variant_str / "relaxation" / "CONTCAR")


def get_pair_arrays(npairs, combtypes):
    return {
        "Fe-Fe": npairs[:, combtypes.index("Fe-Fe")] if "Fe-Fe" in combtypes else 0,
        "Cr-Fe": npairs[:, combtypes.index("Cr-Fe")] if "Cr-Fe" in combtypes else 0,
        "Cr-Cr": npairs[:, combtypes.index("Cr-Cr")] if "Cr-Cr" in combtypes else 0,
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
def plot_neighbor_sro(variant_strings, relax_dir, cutoff, number_of_NN):
    figSRO, axSRO = plt.subplots(2, 2, figsize=(10, 10))

    for variant_str in variant_strings:
        formula, cell, basis = load_variant_cell(relax_dir, variant_str)
        _, c_fe, c_cr, _ = parse_formula(formula)

        distances, npairs, combtypes = pa(formula, cutoff, basis, cell)
        distances, npairs = distances[:number_of_NN], npairs[:number_of_NN]

        pair_arrays = get_pair_arrays(npairs, combtypes)
        N_fe_fe = sum(pair_arrays["Fe-Fe"])
        N_fe_cr = sum(pair_arrays["Cr-Fe"])
        N_cr_cr = sum(pair_arrays["Cr-Cr"])

        N_fe = N_fe_fe + N_fe_cr
        N_cr = N_cr_cr + N_fe_cr

        p_fe_fe = N_fe_fe / N_fe
        p_fe_cr = N_fe_cr / N_fe
        p_cr_cr = N_cr_cr / N_cr
        p_cr_fe = N_fe_cr / N_cr

        a_fe_cr = 1 - p_fe_cr / c_cr
        a_cr_fe = 1 - p_cr_fe / c_fe
        a_fe_fe = 1 - p_fe_fe / c_fe
        a_cr_cr = 1 - p_cr_cr / c_cr

        axSRO[0][0].plot([c_cr], [a_fe_fe], "ko", alpha=0.5, markerfacecolor="none")
        axSRO[1][0].plot([c_cr], [a_cr_fe], "ko", alpha=0.5, markerfacecolor="none")
        axSRO[0][1].plot([c_cr], [a_fe_cr], "ko", alpha=0.5, markerfacecolor="none")
        axSRO[1][1].plot([c_cr], [a_cr_cr], "ko", alpha=0.5, markerfacecolor="none")

    axSRO[0][0].set_xlabel(r"$x_{Cr}$")
    axSRO[0][0].set_ylabel(r"$a_{FeFe}$")
    axSRO[0][0].set_title(r"$Fe-Fe$")

    axSRO[1][0].set_xlabel(r"$x_{Cr}$")
    axSRO[1][0].set_ylabel(r"$a_{CrFe}$")
    axSRO[1][0].set_title(r"$Cr-Fe$")

    axSRO[0][1].set_xlabel(r"$x_{Cr}$")
    axSRO[0][1].set_ylabel(r"$a_{FeCr}$")
    axSRO[0][1].set_title(r"$Fe-Cr$")

    axSRO[1][1].set_xlabel(r"$x_{Cr}$")
    axSRO[1][1].set_ylabel(r"$a_{CrCr}$")
    axSRO[1][1].set_title(r"$Cr-Cr$")

    for ax in axSRO.flatten():
        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)


def plot_pair_probability_sro(variant_strings, relax_dir, cutoff, number_of_NN, NATOMS):
    figSRO, axSRO = plt.subplots(2, 2, figsize=(10, 10))

    for variant_str in variant_strings:
        formula, cell, basis = load_variant_cell(relax_dir, variant_str)
        types_list, c_fe_frac, c_cr, n_fe = parse_formula(formula)

        # Preserve original script behavior:
        # original code overwrote c_Fe with the integer number of Fe atoms.
        c_fe = types_list.count("Fe")
        nal = c_fe

        distances, npairs, combtypes = pa(formula, cutoff * 2, basis, cell)
        distances, npairs = distances[:number_of_NN], npairs[:number_of_NN]

        fe_fe_pairs, fe_cr_pairs, cr_cr_pairs = get_pair_totals(npairs, combtypes, nal, NATOMS)
        total_pairs = fe_fe_pairs + fe_cr_pairs + cr_cr_pairs

        P_fe_fe = fe_fe_pairs / total_pairs
        P_fe_cr = fe_cr_pairs / total_pairs
        P_cr_cr = cr_cr_pairs / total_pairs

        alpha_fe_fe = 1 - (P_fe_fe / (c_fe ** 2)) if c_fe != 0 else float("nan")
        alpha_fe_cr = 1 - (P_fe_cr / (2 * c_fe * c_cr)) if (c_fe != 0 and c_cr != 0) else float("nan")
        alpha_cr_cr = 1 - (P_cr_cr / (c_cr ** 2)) if c_cr != 0 else float("nan")

        axSRO[0][0].plot([c_cr], [alpha_fe_fe], "ko", alpha=0.5, markerfacecolor="none")
        axSRO[1][0].plot([c_cr], [alpha_fe_cr], "ko", alpha=0.5, markerfacecolor="none")
        axSRO[0][1].plot([c_cr], [alpha_fe_cr], "ko", alpha=0.5, markerfacecolor="none")
        axSRO[1][1].plot([c_cr], [alpha_cr_cr], "ko", alpha=0.5, markerfacecolor="none")

    axSRO[0][0].set_xlabel(r"$x_{Cr}$")
    axSRO[0][0].set_ylabel(r"$\alpha_{FeFe}$")
    axSRO[0][0].set_title(r"$Fe-Fe$")

    axSRO[1][0].set_xlabel(r"$x_{Cr}$")
    axSRO[1][0].set_ylabel(r"$\alpha_{CrFe}$")
    axSRO[1][0].set_title(r"$Cr-Fe$")

    axSRO[0][1].set_xlabel(r"$x_{Cr}$")
    axSRO[0][1].set_ylabel(r"$\alpha_{FeCr}$")
    axSRO[0][1].set_title(r"$Fe-Cr$")

    axSRO[1][1].set_xlabel(r"$x_{Cr}$")
    axSRO[1][1].set_ylabel(r"$\alpha_{CrCr}$")
    axSRO[1][1].set_title(r"$Cr-Cr$")

    x_idwc = np.linspace(0.01, 0.99, 100)

    axSRO[0][1].plot(
        x_idwc,
        [1 - 2 * min([ci, (1 - ci)]) / (2 * ci * (1 - ci)) for ci in x_idwc],
        "--",
        alpha=0.5,
    )
    axSRO[1][0].plot([0, 1], [0, 0], "r--", alpha=0.5)
    axSRO[1][0].plot(
        x_idwc,
        [1 - 2 * min([ci, (1 - ci)]) / (2 * ci * (1 - ci)) for ci in x_idwc],
        "--",
        alpha=0.5,
    )

    axSRO[0][1].plot([0, 1], [0, 0], "r--", alpha=0.5)
    axSRO[0][0].plot([0, 1], [0, 0], "r--", alpha=0.5)
    axSRO[1][1].plot([0, 1], [0, 0], "r--", alpha=0.5)

    axSRO[0][0].plot([0, 1], [1, 1], "b--", alpha=0.5)
    axSRO[1][1].plot([0, 1], [1, 1], "b--", alpha=0.5)

    for ax in axSRO.flatten():
        ax.set_xlim(0, 1)


def plot_pair_fraction_and_average_distance(variant_strings, relax_dir, cutoff, number_of_NN, NATOMS):
    figSRO, axSRO = plt.subplots(2, 2, figsize=(10, 10))
    fig_av_iterdistance, ax_av_iterdistance = plt.subplots()

    avd_fe = 0
    avd_cr = 0
    avd_list_x = []
    avd_list_y = []

    for variant_str in variant_strings:
        formula, cell, basis = load_variant_cell(relax_dir, variant_str)
        _, c_fe, c_cr, n_fe = parse_formula(formula)

        distances, npairs, combtypes = pa(formula, cutoff, basis, cell)
        distances, npairs = distances[:number_of_NN], npairs[:number_of_NN]

        N = npairs.sum(axis=1)
        average_distance = np.dot(distances, N) / N.sum()

        avd_list_x.append(c_cr)
        avd_list_y.append(average_distance)

        fe_fe_pairs, fe_cr_pairs, cr_cr_pairs = get_pair_totals(npairs, combtypes, n_fe, NATOMS)

        if n_fe == 0:
            avd_cr = average_distance
        elif n_fe == NATOMS:
            avd_fe = average_distance

        total_pairs = fe_fe_pairs + fe_cr_pairs + cr_cr_pairs
        x_fe_fe = fe_fe_pairs / total_pairs
        x_cr_cr = cr_cr_pairs / total_pairs
        x_fe_cr = fe_cr_pairs / total_pairs

        axSRO[0][0].plot([c_cr], [x_fe_fe], "ko", alpha=0.5, markerfacecolor="none")
        axSRO[1][0].plot([c_cr], [x_fe_cr], "ko", alpha=0.5, markerfacecolor="none")
        axSRO[0][1].plot([c_cr], [x_fe_cr], "ko", alpha=0.5, markerfacecolor="none")
        axSRO[1][1].plot([c_cr], [x_cr_cr], "ko", alpha=0.5, markerfacecolor="none")

    axSRO[0][0].set_xlabel(r"$x_{Cr}$", fontsize=12)
    axSRO[0][0].set_ylabel(r"$x_{FeFe}$", fontsize=12)
    axSRO[0][0].set_title(r"$Fe-Fe$", y=0.99)

    axSRO[1][0].set_xlabel(r"$x_{Cr}$", fontsize=12)
    axSRO[1][0].set_ylabel(r"$x_{CrFe}$", fontsize=12)
    axSRO[1][0].set_title(r"$Cr-Fe$", y=0.99)

    axSRO[0][1].set_xlabel(r"$x_{Cr}$", fontsize=12)
    axSRO[0][1].set_ylabel(r"$x_{FeCr}$", fontsize=12)
    axSRO[0][1].set_title(r"$Fe-Cr$", y=0.99)

    axSRO[1][1].set_xlabel(r"$x_{Cr}$", fontsize=12)
    axSRO[1][1].set_ylabel(r"$x_{CrCr}$", fontsize=12)
    axSRO[1][1].set_title(r"$Cr-Cr$", y=0.99)

    for ax in axSRO.flatten():
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
        ax.axhline(y=1, color="k", linestyle="--", linewidth=0.5)

    x_cr_ideal = np.linspace(0, 1, 50)
    x_fe_fe_ideal = [(1 - x) ** 2 for x in x_cr_ideal]
    x_cr_cr_ideal = [x ** 2 for x in x_cr_ideal]
    x_fe_cr_ideal = [2 * x * (1 - x) for x in x_cr_ideal]

    axSRO[0][0].plot(x_cr_ideal, x_fe_fe_ideal, "r--", alpha=0.5)
    axSRO[0][1].plot(x_cr_ideal, x_fe_cr_ideal, "r--", alpha=0.5)
    axSRO[1][0].plot(x_cr_ideal, x_fe_cr_ideal, "r--", alpha=0.5)
    axSRO[1][1].plot(x_cr_ideal, x_cr_cr_ideal, "r--", alpha=0.5)

    av_d_ideal = [(1 - x) * avd_fe + x * avd_cr for x in x_cr_ideal]
    ax_av_iterdistance.plot(avd_list_x, avd_list_y, "ko", alpha=0.5, markerfacecolor="none")
    ax_av_iterdistance.plot(x_cr_ideal, av_d_ideal, "k-", alpha=0.5, label="ideal")
