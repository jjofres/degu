from pathlib import Path
import os
import re

import numpy as np
# from matplotlib import pyplot as plt
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
