import os
import re
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from coypus.datacontainer import DataContainer as DC
from debyetools.aux_functions import load_cell
from debyetools.pairanalysis import pair_analysis as pa


cmap = plt.colormaps["rainbow"]
LEFT_ELEMENT = "Fe"
RIGHT_ELEMENT = "Cr"


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


def get_valid_temperatures(table):
    return [float(Ti) for Ti in table.T if Ti is not None]


def get_temperature_norm(table):
    temperatures = get_valid_temperatures(table)
    return mcolors.Normalize(vmin=min(temperatures), vmax=max(temperatures))


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


def split_variant_column_key(key):
    parts = key.split("_")
    n_fe = int(parts[2])
    label = "_".join(parts[-2:])
    return n_fe, label


def extract_xy_per_temperature(table, natoms):
    return extract_xy_per_temperature_for_prefix(table, natoms, prefix=None)


def extract_xy_per_temperature_for_prefix(table, natoms, prefix=None):
    valid_temperatures = get_valid_temperatures(table)

    xy_per_temp = {
        f"{Ti:.5f}": {"x": [], "y": [], "label": []}
        for Ti in valid_temperatures
    }

    for row in table:
        if not row.T or row.T[0] is None:
            continue

        temp_key = f"{float(row.T[0]):.5f}"

        for key in row.keys()[2:]:
            if prefix is not None and not key.startswith(prefix):
                continue

            value = row[key][0]
            if value is None:
                continue

            n_fe, label = split_variant_column_key(key)

            xy_per_temp[temp_key]["x"].append((natoms - n_fe) / natoms)
            xy_per_temp[temp_key]["y"].append(float(value))
            xy_per_temp[temp_key]["label"].append(label)

    return xy_per_temp


def annotate_points(ax, x, y, labels, dx=15, dy=0, fontsize=6):
    for label, xi, yi in zip(labels, x, y):
        ax.annotate(
            label,
            xy=(xi, yi),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            fontsize=fontsize,
        )


def plot_temperature_legend_entry(ax, color, temperature):
    ax.plot(
        [],
        [],
        color=color,
        marker="o",
        markerfacecolor="none",
        linestyle="none",
        label=f"T={float(temperature):.0f}",
    )


# ---------------------------
# Data loading / table building
# ---------------------------
def load_tprops_by_variant(tprops, variant_map):
    tprops_all_dict = {}

    for s, sx_list in variant_map.items():
        tprops_all_dict[s] = {}

        for sx in sx_list:
            tprops_all_dict[s][sx] = (
                tprops
                .filter_by_value("sname", f"{s}/{sx}")
                .to_dict()
            )

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

            variant_id = f"{s}_{sx}"

            add_padded_attribute(table_tgs, f"G_{variant_id}", G_i, ntemps)
            add_padded_attribute(table_tgs, f"S_{variant_id}", S_i, ntemps)
            add_padded_attribute(table_tgs, f"H_{variant_id}", H_i, ntemps)
            add_padded_attribute(table_tgs, f"V_{variant_id}", V_i, ntemps)

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

    output_tables = {
        "DHmix": new_temperature_table(T),
        "DVmix": new_temperature_table(T),
        "DHform": new_temperature_table(T),
    }

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

            values_by_name = {
                "DHmix": H_fecr - n_cr / natoms * H_cr - n_fe / natoms * H_fe,
                "DVmix": V_fecr - n_cr / natoms * V_cr - n_fe / natoms * V_fe,
                "DHform": H_fecr - n_cr / natoms * Href_cr - n_fe / natoms * Href_fe,
            }

            for name, values in values_by_name.items():
                add_padded_attribute(
                    output_tables[name],
                    f"{name}_{variant_id}",
                    values,
                    ntemps,
                )

    output_tables["DHmix"].save(str(hmix_outpath))
    output_tables["DVmix"].save(str(vmix_outpath))
    output_tables["DHform"].save(str(hform_outpath))

    return output_tables["DHmix"], output_tables["DVmix"], output_tables["DHform"]


# ---------------------------
# Convex hull / volume plots
# ---------------------------
def scale_mix_values(values, typehv):
    values = np.array(values, dtype=float)

    if typehv == "V":
        return values / 1e-5

    if typehv in {"H", "Hf"}:
        return values / 1000

    raise ValueError(f"Unknown typehv: {typehv!r}")


def format_hv_mix_axis(ax, typehv):
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$x_{Cr}$ (molar)")

    if typehv == "H":
        ax.set_ylabel(r"$\Delta H_{mix}$ (kJ/mol-atom)")
        ax.axhline(0, color="black", lw=0.5, ls="--")
    elif typehv == "V":
        ax.set_ylabel(r"$\Delta V_{mix}$ ($10e^{-5}m^3$/mol-atom)")
        ax.set_ylim(-0.1, 0.1)
        ax.axhline(0, color="black", lw=0.5, ls="--")
    elif typehv == "Hf":
        ax.set_ylabel(r"$\Delta H_{f}$ (kJ/mol-atom)")
    else:
        raise ValueError(f"Unknown typehv: {typehv!r}")

    ax.legend(fontsize=6)


def plot_HV_mix(ax, table_Hmix, totnats, typehv):
    xy_per_temp = extract_xy_per_temperature(table_Hmix, totnats)
    norm = get_temperature_norm(table_Hmix)

    last_x = []
    last_y = []
    last_labels = []

    for Ti, data in xy_per_temp.items():
        x = data["x"]
        y = scale_mix_values(data["y"], typehv)
        color = cmap(norm(float(Ti)))

        ax.plot(x, y, "o", label=f"T={float(Ti):.0f}", color=color, alpha=0.3,)

        last_x = x
        last_y = y
        last_labels = data["label"]

    annotate_points(ax, last_x, last_y, last_labels)
    format_hv_mix_axis(ax, typehv)


def scatter_labeled_points(ax, x, y, labels, color):
    markers = [f"${label.split('_')[1]}$" for label in labels]

    for xi, yi, marker in zip(x, y, markers):
        ax.scatter(xi, yi, marker=marker, edgecolors="none", facecolors=color, )


def plot_vegard_line(ax, xs, ys):
    indices = np.argsort(xs)
    xs = np.array(xs)[indices]
    ys = np.array(ys)[indices]

    min_v = ys[0]
    max_v = ys[-1]
    vegard_y = xs * max_v + (1 - xs) * min_v

    ax.plot(xs, vegard_y, "k--", linewidth=0.5)


def format_vmix_vegard_axes(amix, ay, limitsVm, limitsV, title_temperature=None):
    amix.set_ylim(*limitsVm)
    ay.set_ylim(*limitsV)

    amix.set_xlim(0, 1)
    ay.set_xlim(0, 1)

    if title_temperature is not None:
        amix.set_title(f"T = {float(title_temperature):.2f} K")

    amix.set_xlabel(r"$x_{Cr}$ (molar)")
    ay.set_xlabel(r"$x_{Cr}$ (molar)")

    amix.set_ylabel(r"$\Delta V_{mix}$ ($10e^{-5}m^3$/mol-atom)")
    ay.set_ylabel(r"$V$ ($10e^{-5}m^3$/mol-atom)")

    amix.axhline(0, color="black", lw=0.5, ls="--")

    amix.legend(fontsize=6, ncols=2)
    ay.legend(fontsize=6, ncols=1)


def plot_vmix_convex_hulls_and_vegard(
    axxx,
    table_vmix,
    table_tgs,
    NATOMS,
    limitsVm=(-0.02, 0.02),
    limitsV=(0.6, 0.8),
):
    xy_vmix = extract_xy_per_temperature(table_vmix, NATOMS)
    xy_volume = extract_xy_per_temperature_for_prefix(table_tgs, NATOMS, prefix="V_")

    ay, amix = axxx[0], axxx[1]
    norm = get_temperature_norm(table_vmix)

    last_mix_x = []
    last_mix_y = []
    last_mix_labels = []
    last_temperature = None

    for i, (Ti, mix_data) in enumerate(xy_vmix.items()):
        color = cmap(norm(float(Ti)))

        xm = mix_data["x"]
        ym = np.array(mix_data["y"]) / 1e-5

        scatter_labeled_points(amix, xm, ym, mix_data["label"], color)
        plot_temperature_legend_entry(amix, color, Ti)

        last_mix_x = xm
        last_mix_y = ym
        last_mix_labels = mix_data["label"]
        last_temperature = Ti

        if i % 3 != 0:
            continue

        volume_data = xy_volume[Ti]
        xv = np.array(volume_data["x"])
        yv = np.array(volume_data["y"]) / 1e-5

        sort_ix = np.argsort(xv)
        xv = xv[sort_ix]
        yv = yv[sort_ix]
        labels = np.array(volume_data["label"])[sort_ix]

        scatter_labeled_points(ay, xv, yv, labels, color)
        plot_temperature_legend_entry(ay, color, Ti)
        plot_vegard_line(ay, xv, yv)

    annotate_points(amix, last_mix_x, last_mix_y, last_mix_labels)
    format_vmix_vegard_axes(amix, ay, limitsVm, limitsV, title_temperature=last_temperature)


# ---------------------------
# Pair-analysis helpers
# ---------------------------
@dataclass
class VariantPairStats:
    variant_str: str
    c_left: float
    c_right: float
    distances: np.ndarray
    npairs: np.ndarray
    n_left_left: float
    n_left_right: float
    n_right_right: float

    @property
    def total_pairs(self):
        return self.n_left_left + self.n_left_right + self.n_right_right

    @property
    def n_left_neighbors(self):
        return 2 * self.n_left_left + self.n_left_right

    @property
    def n_right_neighbors(self):
        return 2 * self.n_right_right + self.n_left_right


def load_variant_cell(relax_dir, variant_str):
    return load_cell(relax_dir / variant_str / "relaxation" / "CONTCAR")


def parse_type_concentrations(types_str):
    types_list = re.findall(r"[A-Z][a-z]*", types_str)
    ntypes = len(types_list)

    return {
        "types_list": types_list,
        "c_left": types_list.count("Fe") / ntypes,
        "c_right": types_list.count("Cr") / ntypes,
    }


def get_pair_column(npairs, combtypes, pair_name, fallback_pair_name=None):
    if pair_name in combtypes:
        return npairs[:, combtypes.index(pair_name)]

    if fallback_pair_name is not None and fallback_pair_name in combtypes:
        return npairs[:, combtypes.index(fallback_pair_name)]

    return np.zeros(len(npairs))


def find_npairs_ix(npairs, np_sum):
    for i in range(1, len(npairs)):
        current_sum = sum(np.sum(npairs[0:i], axis=1))
        if abs(current_sum - np_sum) < 0.5:
            return i

    return len(npairs)

def get_variant_cell_data(variant_str, crystal_data_dict):
    types_str_dict, cell_dict, basis_dict = crystal_data_dict
    return (
        types_str_dict[variant_str],
        cell_dict[variant_str],
        basis_dict[variant_str],
    )
def get_variant_pair_stats(variant_str, cutoff, n_pairs_ideal, crystal_data_dict):
    types_str, cell, basis = get_variant_cell_data(variant_str, crystal_data_dict)
    concentrations = parse_type_concentrations(types_str)

    distances, npairs, combtypes = pa(types_str, cutoff, basis, cell)

    number_of_nn = find_npairs_ix(npairs, n_pairs_ideal)
    distances = distances[:number_of_nn]
    npairs = npairs[:number_of_nn]

    left_left_name = f"{LEFT_ELEMENT}-{LEFT_ELEMENT}"
    left_right_name = f"{LEFT_ELEMENT}-{RIGHT_ELEMENT}"
    right_left_name = f"{RIGHT_ELEMENT}-{LEFT_ELEMENT}"
    right_right_name = f"{RIGHT_ELEMENT}-{RIGHT_ELEMENT}"

    npairs_left_left = get_pair_column(npairs, combtypes, left_left_name)
    npairs_left_right = get_pair_column(
        npairs,
        combtypes,
        right_left_name,
        fallback_pair_name=left_right_name,
    )
    npairs_right_right = get_pair_column(npairs, combtypes, right_right_name)

    return VariantPairStats(
        variant_str=variant_str,
        c_left=concentrations["c_left"],
        c_right=concentrations["c_right"],
        distances=distances,
        npairs=npairs,
        n_left_left=sum(npairs_left_left),
        n_left_right=sum(npairs_left_right),
        n_right_right=sum(npairs_right_right),
    )


def iter_variant_pair_stats(variant_strings, cutoff, n_pairs_ideal, crystal_data_dict):
    for variant_str in variant_strings:
        yield get_variant_pair_stats(
            variant_str,
            cutoff,
            n_pairs_ideal,
            crystal_data_dict,
        )



def safe_divide(numerator, denominator):
    return numerator / denominator if denominator > 0 else np.nan


# ---------------------------
# SRO / pair calculations
# ---------------------------
def compute_average_pair_distance(stats):
    pair_counts_by_distance = stats.npairs.sum(axis=1)
    return np.dot(stats.distances, pair_counts_by_distance) / pair_counts_by_distance.sum()


def compute_pair_fractions(stats):
    total = stats.total_pairs

    return {
        "FeFe": safe_divide(stats.n_fe_fe, total),
        "FeCr": safe_divide(stats.n_fe_cr, total),
        "CrFe": safe_divide(stats.n_fe_cr, total),
        "CrCr": safe_divide(stats.n_cr_cr, total),
    }


def compute_alpha_sro(stats):
    pair_fractions = compute_pair_fractions(stats)

    alpha_fe_fe = (
        1 - pair_fractions["FeFe"] / stats.c_fe**2
        if stats.c_fe > 0
        else np.nan
    )
    alpha_fe_cr = (
        1 - pair_fractions["FeCr"] / (2 * stats.c_fe * stats.c_cr)
        if stats.c_fe > 0 and stats.c_cr > 0
        else np.nan
    )
    alpha_cr_cr = (
        1 - pair_fractions["CrCr"] / stats.c_cr**2
        if stats.c_cr > 0
        else np.nan
    )

    return {
        "FeFe": alpha_fe_fe,
        "FeCr": alpha_fe_cr,
        "CrFe": alpha_fe_cr,
        "CrCr": alpha_cr_cr,
    }


def compute_directional_sro(stats):
    n_fe = stats.n_fe_neighbors
    n_cr = stats.n_cr_neighbors

    p_fe_fe = safe_divide(2 * stats.n_fe_fe, n_fe)
    p_fe_cr = safe_divide(stats.n_fe_cr, n_fe)
    p_cr_cr = safe_divide(2 * stats.n_cr_cr, n_cr)
    p_cr_fe = safe_divide(stats.n_fe_cr, n_cr)

    return {
        "FeFe": 1 - p_fe_fe / stats.c_fe if n_fe > 0 and stats.c_fe > 0 else np.nan,
        "FeCr": 1 - p_fe_cr / stats.c_cr if n_fe > 0 and stats.c_cr > 0 else np.nan,
        "CrFe": 1 - p_cr_fe / stats.c_fe if n_cr > 0 and stats.c_fe > 0 else np.nan,
        "CrCr": 1 - p_cr_cr / stats.c_cr if n_cr > 0 and stats.c_cr > 0 else np.nan,
    }


# ---------------------------
# SRO / pair plotting helpers
# ---------------------------
SRO_AXES = {
    "FeFe": (0, 0),
    "CrFe": (1, 0),
    "FeCr": (0, 1),
    "CrCr": (1, 1),
}


def plot_sro_point_grid(ax_grid, stats, values, label_prefix):
    for pair_name, value in values.items():
        i, j = SRO_AXES[pair_name]
        ax_grid[i][j].plot(
            [stats.c_cr],
            [value],
            "ko",
            label=f"{stats.variant_str}_{label_prefix}_{pair_name}",
            alpha=0.5,
            markerfacecolor="none",
        )


def add_random_pair_fraction_lines(ax_grid):
    x = np.linspace(0, 1, 50)

    ax_grid[0][0].plot(x, (1 - x) ** 2, "r--", alpha=0.5)
    ax_grid[0][1].plot(x, 2 * x * (1 - x), "r--", alpha=0.5)
    ax_grid[1][0].plot(x, 2 * x * (1 - x), "r--", alpha=0.5)
    ax_grid[1][1].plot(x, x**2, "r--", alpha=0.5)


def add_alpha_reference_lines(ax_grid):
    x = np.linspace(0.01, 0.99, 100)
    ordered_cross = [
        1 - 2 * min(ci, 1 - ci) / (2 * ci * (1 - ci))
        for ci in x
    ]

    ax_grid[0][1].plot(x, ordered_cross, "--", label="ordered", alpha=0.5)
    ax_grid[1][0].plot(x, ordered_cross, "--", label="ordered", alpha=0.5)

    ax_grid[0][1].plot([0, 1], [0, 0], "r--", label="random", alpha=0.5)
    ax_grid[1][0].plot([0, 1], [0, 0], "r--", label="random", alpha=0.5)
    ax_grid[0][0].plot([0, 1], [0, 0], "r--", label="random", alpha=0.5)
    ax_grid[1][1].plot([0, 1], [0, 0], "r--", label="random", alpha=0.5)

    ax_grid[0][0].plot([0, 1], [1, 1], "b--", label="ordered", alpha=0.5)
    ax_grid[1][1].plot([0, 1], [1, 1], "b--", label="ordered", alpha=0.5)


def set_grid_ylim(ax_grid, lims_y):
    if lims_y is None:
        return

    for ax in ax_grid.flatten():
        ax.set_ylim(*lims_y)


def format_a_sro_axes(ax_grid):
    labels = {
        "FeFe": (r"$a_{FeFe}$", r"$Fe-Fe$"),
        "CrFe": (r"$a_{CrFe}$", r"$Cr-Fe$"),
        "FeCr": (r"$a_{FeCr}$", r"$Fe-Cr$"),
        "CrCr": (r"$a_{CrCr}$", r"$Cr-Cr$"),
    }

    for pair_name, (ylabel, title) in labels.items():
        i, j = SRO_AXES[pair_name]
        ax = ax_grid[i][j]
        ax.set_xlabel(r"$x_{Cr}$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)

    ax_grid[0][0].axhline(y=1, color="k", linestyle="--", linewidth=0.5)
    ax_grid[1][1].axhline(y=1, color="k", linestyle="--", linewidth=0.5)
    ax_grid[1][0].axhline(y=-1, color="k", linestyle="--", linewidth=0.5)
    ax_grid[0][1].axhline(y=-1, color="k", linestyle="--", linewidth=0.5)


# ---------------------------
# Public SRO / pair plots
# ---------------------------
def plot_avd_sro(ax_av_iterdistance, variant_strings, cutoff, n_pairs_ideal, crystal_data_dict,):
    avd_left = 0
    avd_right = 0
    avd_list_x = []
    avd_list_y = []
    last_variant_str = ""

    for stats in iter_variant_pair_stats( variant_strings, cutoff, n_pairs_ideal, crystal_data_dict, ):
        average_distance = compute_average_pair_distance(stats)

        avd_list_x.append(stats.c_right)
        avd_list_y.append(average_distance)
        last_variant_str = stats.variant_str

        if stats.c_left == 0:
            avd_right = average_distance
        if stats.c_left == 1:
            avd_left = average_distance

    x_ideal = np.linspace(0, 1, 50)
    avd_ideal = [(1 - x) * avd_left + x * avd_right for x in x_ideal]

    ax_av_iterdistance.plot(avd_list_x, avd_list_y, "ko", label=f"{last_variant_str}_average_distance", alpha=0.5, markerfacecolor="none", )

    ax_av_iterdistance.plot(x_ideal, avd_ideal, "k-", alpha=0.5, label="ideal")


def plot_x_sro(axSRO3, variant_strings, RELAX_DIR, cutoff, n_pairs_ideal):
    for stats in iter_variant_pair_stats(variant_strings, RELAX_DIR, cutoff, n_pairs_ideal):
        plot_sro_point_grid(
            axSRO3,
            stats,
            compute_pair_fractions(stats),
            label_prefix="x",
        )

    add_random_pair_fraction_lines(axSRO3)


def plot_alpha_sro(axSRO2, variant_strings, RELAX_DIR, cutoff, n_pairs_ideal, lims_y=None, ):
    for stats in iter_variant_pair_stats(variant_strings, RELAX_DIR, cutoff, n_pairs_ideal):
        plot_sro_point_grid(axSRO2, stats, compute_alpha_sro(stats), label_prefix="alpha", )

    add_alpha_reference_lines(axSRO2)
    set_grid_ylim(axSRO2, lims_y)


def plot_a_sro(axSRO, variant_strings, RELAX_DIR, cutoff, n_pairs_ideal):
    for stats in iter_variant_pair_stats(variant_strings, RELAX_DIR, cutoff, n_pairs_ideal):
        plot_sro_point_grid(axSRO, stats, compute_directional_sro(stats), label_prefix="a", )

    format_a_sro_axes(axSRO)
