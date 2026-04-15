import re
from typing import List, Dict, Any

import numpy as np


def extract_convergence_data(log_path: str) -> List[Dict[str, Any]]:
    header_re = re.compile(r'^\s*(\d+):\s+(\S+)')
    encut_k_re = re.compile(r'calcs_conv_(\d+)_([0-9.]+)/')
    struct_re = re.compile(r'/s_(\d+)/sx_(\d+)/')

    etot_re = re.compile(r'\bTOTEN_eV\b[^-+0-9]*([-+]?\d+\.\d+(?:[Ee][+-]?\d+)?)')
    natoms_re = re.compile(r'\bNIONS\b[^0-9]*([0-9]+)')
    cpu_re = re.compile(r'\boutcar_total_cpu_time_sec\b[^0-9]*([0-9.]+)')

    outcar_incomplete_re = re.compile(r'\bOUTCAR_INCOMPLETE\b')

    results: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None

    # 👇 key change: specify encoding and be tolerant to weird bytes
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            m_header = header_re.match(line)
            if m_header:
                if current is not None:
                    results.append(current)

                idx_str, path = m_header.groups()
                current = {
                    "ix": int(idx_str),
                    "path": path,
                }

                m_ek = encut_k_re.search(path)
                if m_ek:
                    current["ENCUT"] = float(m_ek.group(1))
                    current["KSPACING"] = float(m_ek.group(2))

                m_struct = struct_re.search(path)
                if m_struct:
                    current["s"] = int(m_struct.group(1))
                    current["sx"] = int(m_struct.group(2))

                continue

            if current is None:
                continue

            m_etot = etot_re.search(line)
            if m_etot:
                current["Etot"] = float(m_etot.group(1))

            m_nat = natoms_re.search(line)
            if m_nat:
                current["Natoms"] = int(m_nat.group(1))

            m_cpu = cpu_re.search(line)
            if m_cpu:
                current["CPU_time_s"] = float(m_cpu.group(1))

            m_outcar_inc = outcar_incomplete_re.search(line)
            if m_outcar_inc:
                current['CPU_time_s'] = None  # Mark as incomplete

    if current is not None:
        results.append(current)

    return results


def list_of_dicts_to_dict_of_lists(data: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    result: Dict[str, List[Any]] = {}
    for entry in data:
        for key, value in entry.items():
            if key not in result:
                result[key] = []
            result[key].append(value)
    return result


def get_cpu_etot(ecut, ks, datacontainer_log, s, sx,str_spec='Ca-Sr'):
    dc2 = (datacontainer_log
           .filter_by_value("KSPACING", ks)
           .filter_by_value("ENCUT", ecut)
           .filter_by_value("path", f"calcs_conv_{ecut}_{ks:.2f}/{str_spec}/{s}/{sx}/relaxation"))

    if len(dc2["CPU_time_s"]) == 0:
        return None, None
    else:
        return dc2["CPU_time_s"][0], dc2["Etot"][0]


from coypus.datacontainer import DataContainer as dc


def get_data4plot(filename, encuts, kspacings, s, sx, str_spec = 'Ca-Sr'):
    rows = extract_convergence_data(filename)
    rows_dict = list_of_dicts_to_dict_of_lists(rows)
    datacontainer_log = dc(rows_dict)


    # compute all combinations once
    vals = {
        (ecut, ks): get_cpu_etot(ecut, ks, datacontainer_log, s, sx, str_spec=str_spec)
        for ecut in encuts
        for ks in kspacings
    }

    data4plot_cpu_ks = {ks: [vals[(ecut, ks)][0] for ecut in encuts] for ks in kspacings}
    data4plot_etot_ks = {ks: [vals[(ecut, ks)][1] for ecut in encuts] for ks in kspacings}
    data4plot_cpu_ecut = {ecut: [vals[(ecut, ks)][0] for ks in kspacings] for ecut in encuts}
    data4plot_etot_ecut = {ecut: [vals[(ecut, ks)][1] for ks in kspacings] for ecut in encuts}

    return data4plot_cpu_ks, data4plot_etot_ks, data4plot_cpu_ecut, data4plot_etot_ecut


from matplotlib import pyplot as plt


def plot_convtest(data4plot, encuts, kspacings, fig, ax):
    data4plot_cpu_ks, data4plot_etot_ks, data4plot_cpu_ecut, data4plot_etot_ecut = data4plot
    colors = ['r', 'b', 'k', 'c', 'm', 'y', 'k']
    for ix, ks in enumerate(kspacings):
        ax[0, 0].plot(encuts, data4plot_cpu_ks[ks], marker='o', label=f'KSPACING={ks}', color=colors[ix % len(colors)])
        ax[0, 1].plot(encuts, data4plot_etot_ks[ks], marker='o', label=f'KSPACING={ks}', color=colors[ix % len(colors)])

    for ix, ecut in enumerate(encuts):
        ax[1, 0].plot(kspacings, data4plot_cpu_ecut[ecut], marker='o', label=f'ENCUT={ecut}',
                      color=colors[ix % len(colors)])
        ax[1, 1].plot(kspacings, data4plot_etot_ecut[ecut], marker='o', label=f'ENCUT={ecut}',
                      color=colors[ix % len(colors)])

    for i in range(2):
        ax[0, i].set_xlabel('ENCUT')
        ax[1, i].set_xlabel('KSPACING')
        ax[i, 0].set_ylabel('CPU Time (s)')
        ax[i, 1].set_ylabel('Total Energy (eV)')

    for i in range(2):
        for j in range(2):
            ax[i, j].legend()

    plt.tight_layout()

    return fig, ax


import re
from collections import defaultdict
from typing import Dict, List, Union


def sort_EvV_by_volume(EvV_data):
    sorted_data = {}

    for struct, vals in EvV_data.items():
        vols = vals.get("final_volume_A3", [])
        Es = vals.get("TOTEN", [])
        Ns = vals.get("NIONS", [])
        mass = vals.get("total_mass_g", [])
        count_Fe = vals.get("count_Fe", [])
        count_Cr = vals.get("count_Cr", [])
        count_Sr = vals.get("count_Sr", [])

        # If nothing there, just copy as-is
        if not vols:
            sorted_data[struct] = {
                "NIONS": list(Ns),
                "TOTEN": list(Es),
                "final_volume_A3": list(vols),
                "total_mass_g": list(mass),
                "count_Fe": list(count_Fe),
                "count_Cr": list(count_Cr),
                "count_Sr": list(count_Sr),
            }
            continue

        # zip -> sort by volume -> unzip
        triples = sorted(zip(vols, Es, Ns, mass, count_Fe, count_Cr, count_Sr), key=lambda t: t[0])
        sorted_vols, sorted_Es, sorted_Ns, sorted_mass, sorted_count_Fe, sorted_count_Cr, sorted_count_Sr = map(list,
                                                                                                                zip(*triples))

        sorted_data[struct] = {
            "NIONS": sorted_Ns,
            "TOTEN": sorted_Es,
            "final_volume_A3": sorted_vols,
            "total_mass_g": sorted_mass,
            "count_Fe": sorted_count_Fe,
            "count_Cr": sorted_count_Cr,
            "count_Sr": sorted_count_Sr,

        }

    return sorted_data


def extract_EvV(log_path, elastic=False):
    if not elastic:
        # Block header: "378: calcs_EvV/Ca-Sr/s_0/sx_0/100"
        header_re = re.compile(r'^\s*\d+:\s+(\S+)')
        # Extract s_*/sx_* from the path
        struct_re = re.compile(r'/s_(\d+)/sx_(\d+)/')
    else:
        # Block header: "378: calcs_strain_elastic/Ca-Sr/s_0/sx_0/eps1/100"
        header_re = re.compile(r'^\s*\d+:\s+(\S+)')
        # Extract s_*/sx_*/eps* from the path
        struct_re = re.compile(r'/s_(\d+)/sx_(\d+)/eps(\d+)/')

    # Generic float pattern (handles scientific notation too)
    float_pat = r'([-+]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)'

    # Metrics lines inside "Metrics:" block
    nions_re = re.compile(r'\bNIONS:\s*' + float_pat)
    toten_re = re.compile(r'\bTOTEN_eV:\s*' + float_pat)
    vol_re = re.compile(r'\bfinal_volume_A3:\s*' + float_pat)
    mass_re = re.compile(r'\btotal_mass_g:\s*' + float_pat)
    count_Fe_re = re.compile(r'\bcount_Fe:\s*' + float_pat)
    count_Cr_re = re.compile(r'\bcount_Sr:\s*' + float_pat)
    # count_Sr_re = re.compile(r'\bcount_Sr:\s*' + float_pat)

    # data["s_0/sx_0"] = {"NIONS": [...], "TOTEN": [...], "final_volume_A3": [...]}
    data: Dict[str, Dict[str, List[Union[int, float]]]] = defaultdict(
        lambda: {"NIONS": [], "TOTEN": [], "final_volume_A3": [], "total_mass_g": [], "count_Fe": [], "count_Cr": [],
                 # "count_Sr": [],
                 }
    )

    current_struct = None  # type: Union[str, None]
    current_nions = None  # type: Union[int, float, None]
    current_toten = None  # type: Union[float, None]
    current_vol = None  # type: Union[float, None]
    current_mass = None  # type: Union[float, None]
    current_count_Fe = None  # type: Union[float, None]
    current_count_Cr = None  # type: Union[float, None]
    # current_count_Sr = None  # type: Union[float, None]

    def flush_current():
        """If we have a complete set of metrics, store them into data."""
        nonlocal current_struct, current_nions, current_toten, current_vol
        if (
                current_struct is not None
                and current_nions is not None
                and current_toten is not None
                and current_vol is not None
                and current_mass is not None
                and current_count_Fe is not None
                and current_count_Cr is not None
                # and current_count_Sr is not None
        ):
            entry = data[current_struct]
            entry["NIONS"].append(current_nions)
            entry["TOTEN"].append(current_toten)
            entry["final_volume_A3"].append(current_vol)
            entry["total_mass_g"].append(current_mass)
            entry["count_Fe"].append(current_count_Fe)
            entry["count_Cr"].append(current_count_Cr)
            # entry["count_Sr"].append(current_count_Sr)

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # New block?
            m_header = header_re.match(line)
            if m_header:
                # Store the previous block before starting a new one
                flush_current()

                path = m_header.group(1)
                m_struct = struct_re.search(path)
                if m_struct:
                    if not elastic:
                        s_idx, sx_idx = m_struct.groups()
                        current_struct = f"s_{s_idx}/sx_{sx_idx}"
                    else:
                        s_idx, sx_idx, eps_idx = m_struct.groups()
                        current_struct = f"s_{s_idx}/sx_{sx_idx}/eps{eps_idx}"
                else:
                    current_struct = None  # anything without s_/sx_ is ignored

                current_nions = None
                current_toten = None
                current_vol = None
                current_mass = None
                current_count_Fe = None
                current_count_Cr = None
                # current_count_Sr = None
                continue

            # If we don't have a valid s_*/sx_* for this block, skip lines
            if current_struct is None:
                continue

            # Try to catch metrics lines
            m_nions = nions_re.search(line)
            if m_nions:
                # NIONS is really an integer, but it appears as float in the log
                current_nions = int(float(m_nions.group(1)))

            m_toten = toten_re.search(line)
            if m_toten:
                current_toten = float(m_toten.group(1))

            m_vol = vol_re.search(line)
            if m_vol:
                current_vol = float(m_vol.group(1))

            m_mass = mass_re.search(line)
            if m_mass:
                current_mass = float(m_mass.group(1))

            m_count_Fe = count_Fe_re.search(line)
            if m_count_Fe:
                current_count_Fe = float(m_count_Fe.group(1))

            m_count_Cr = count_Cr_re.search(line)
            if m_count_Cr:
                current_count_Cr = float(m_count_Cr.group(1))
            # m_count_Sr = count_Sr_re.search(line)
            # if m_count_Sr:
            #     current_count_Sr = float(m_count_Sr.group(1))

    # Flush last block
    flush_current()

    result = dict(data)
    EvV_data_all = sort_EvV_by_volume(result)

    return EvV_data_all


def extract_EvV_by_structure(EvV_data, structure_path):
    # For example, for s_0/sx_0:
    s0sx0 = EvV_data.get(structure_path, {})
    volumes = s0sx0.get("final_volume_A3", [])
    energies = s0sx0.get("TOTEN", [])
    mass = s0sx0.get("total_mass_g", [])
    natoms = s0sx0.get("NIONS", [])
    # count_Fe = s0sx0.get("count_Fe", [])
    # count_Cr = s0sx0.get("count_Cr", [])
    # count_Sr = s0sx0.get("count_Sr", [])

    volumes = np.array([v / n for v, n in zip(volumes, natoms)]) * (1e-30 * 6.02e23)
    energies = np.array([e / n for e, n in zip(energies, natoms)]) * (0.160218e-18 * 6.02214e23)
    mass = np.array([m / n * 6.02214076e23 for m, n in zip(mass, natoms)])  # grams per atom

    return volumes, energies, mass


def mass_to_g_per_mol(mass_grams, natoms):
    """Convert mass in grams to grams per mole using number of atoms."""
    return mass_grams / natoms * 6.02214076e23


def NfV_function(E, N, Ef):
    ixss = 6
    ixse = -1

    ix_V0 = 4
    EfV0 = float(Ef[ix_V0])
    ixs = [i for i, x in enumerate(E[ix_V0]) if x >= Ef[ix_V0]]
    E1 = float(E[ix_V0][ixs[0] - 1])
    E2 = float(E[ix_V0][ixs[0]])
    N1 = float(N[ix_V0][ixs[0] - 1])
    N2 = float(N[ix_V0][ixs[0]])
    NfV0 = (EfV0 - E1) * (N2 - N1) / (E2 - E1) + N1
    NfV = np.array([NfV0 * np.sqrt(ef / EfV0) for ef in Ef][ixss:ixse])

    return NfV


def print_elastic_results(resall):
    Cij_str = ''
    Eigenvalues_str = ''
    Av_str = ''
    var_str = ''
    dir_str = ''

    Cij_str = Cij_str + 'Cijs:\n'
    Cij_str = Cij_str + 'C_{11} C_{12} C_{13} C_{22} C_{23} C_{33} C_{44} C_{55} C_{66}' + '\n'

    Eigenvalues_str = Eigenvalues_str + 'Eigenvalues' + '\n'
    Eigenvalues_str = Eigenvalues_str + '\\lambda_{1} \\lambda_{2} \\lambda_{3} \\lambda_{4} \\lambda_{5} \\lambda_{6}' + '\n'

    Av_str = Av_str + 'Average properties' + '\n'
    Av_str = Av_str + 'K_{VRH} K_{Voigt} K_{Reuss} E_{VRH} E_{Voigt} E_{Reuss} G_{VRH} G_{Voigt} G_{Reuss} nu_{VRH} nu_{Voigt} nu_{Reuss} A^U G/K' + '\n'

    var_str = var_str + 'Variation of elastic moduli' + '\n'
    var_str = var_str + '     Young\'s modulus     |    Linear compresibility  |        Shear modulus      |     Poisson\'s ratio' + '\n'
    var_str = var_str + 'Min    Max    Anisotropy |  Min    Max    Anisotropy |  Min    Max    Anisotropy |  Min    Max    Anisotropy ' + '\n'

    dir_str = dir_str + 'Directions' + '\n'
    dir_str = dir_str + '     Young\'s modulus     |    Linear compresibility  |        Shear modulus      |     Poisson\'s ratio' + '\n'
    dir_str = dir_str + '   Min   Max     | Min    Max    |  Min    Max    |  Min    Max    ' + '\n'

    for resdata in resall:
        Cij = resdata['Cijs']
        Cij_str = Cij_str + f'{Cij[0, 0]:.2f} {Cij[0, 1]:.2f} {Cij[0, 2]:.2f} {Cij[1, 1]:.2f} {Cij[1, 2]:.2f} {Cij[2, 2]:.2f} {Cij[3, 3]:.2f} {Cij[4, 4]:.2f} {Cij[5, 5]:.2f}' + '\n'

        Eigenvalues_str = Eigenvalues_str + f'{resdata["eigenvalues"][0]:.2f} {resdata["eigenvalues"][1]:.2f} {resdata["eigenvalues"][2]:.2f} {resdata["eigenvalues"][3]:.2f} {resdata["eigenvalues"][4]:.2f} {resdata["eigenvalues"][5]:.2f}' + '\n'

        K_vrh, E_vrh, G_vrh, nu_vrh = resdata['average_properties']['Hill']
        K_voigt, E_voigt, G_voigt, nu_voigt = resdata['average_properties']['Voigt']
        K_reuss, E_reuss, G_reuss, nu_reuss = resdata['average_properties']['Reuss']
        Au = K_voigt / K_reuss + 5 * G_voigt / G_reuss - 6
        Av_str = Av_str + f'{K_vrh:.2f} {K_voigt:.2f} {K_reuss:.2f} {E_vrh:.2f} {E_voigt:.2f} {E_reuss:.2f} {G_vrh:.2f} {G_voigt:.2f} {G_reuss:.2f} {nu_vrh:.2f} {nu_voigt:.2f} {nu_reuss:.2f} {Au:.3f} {G_vrh / K_vrh:.3f}' + '\n'

        E = resdata['variation_properties']['Young']
        L = resdata['variation_properties']['LinearCompressibility']
        G1 = resdata['variation_properties']['Shear']
        G2 = resdata['variation_properties']['Shear2']
        P1 = resdata['variation_properties']['Poisson']
        P2 = resdata['variation_properties']['Poisson2']
        Emin, Emax, Eanis, Emind, Emaxd = E['min'], E['max'], E['anisotropy'], E['min_direction'], E['max_direction']
        Lmin, Lmax, Lanis, Lmind, Lmaxd = L['min'], L['max'], L['anisotropy'], L['min_direction'], L['max_direction']
        G1min, G1max, G1anis, G1mind, G1maxd = G1['min'], G1['max'], G1['anisotropy'], G1['min_direction'], G1[
            'max_direction']
        G2min, G2max, G2anis, G2mind, G2maxd = G2['min'], G2['max'], G2['anisotropy'], G2['min_direction'], G2[
            'max_direction']
        P1min, P1max, P1anis, P1mind, P1maxd = P1['min'], P1['max'], P1['anisotropy'], P1['min_direction'], P1[
            'max_direction']
        P2min, P2max, P2anis, P2mind, P2maxd = P2['min'], P2['max'], P2['anisotropy'], P2['min_direction'], P2[
            'max_direction']

        var_str = var_str + f'{Emin:.2f} {Emax:.2f} {Eanis:.2f} '
        var_str = var_str + f'{Lmin:.3f} {Lmax:.3f} {Lanis:.3f} '
        var_str = var_str + f'{G1min:.2f} {G1max:.2f} {G1anis:.3f} '
        var_str = var_str + f'{P1min:.2f} {P1max:.2f} {P1anis:.3f} ' + '\n'
        var_str = var_str + f'x     x     x         x    x    x      {G2min:.2f} {G2max:.2f} {G2anis:.3f}  '
        var_str = var_str + f'{P2min:.2f} {P2max:.2f} {P2anis:.3f}  ' + '\n'

        dir_str = dir_str + f'{Emind[0]:.2f} {Emaxd[0]:.2f}  '
        dir_str = dir_str + f'{Lmind[0]:.2f} {Lmaxd[0]:.2f}  '
        dir_str = dir_str + f'{G1mind[0]:.2f} {G1maxd[0]:.2f}  '
        dir_str = dir_str + f'{P1mind[0]:.2f} {P1maxd[0]:.2f} ' + '\n'
        dir_str = dir_str + f'{Emind[1]:.2f} {Emaxd[1]:.2f}  '
        dir_str = dir_str + f'{Lmind[1]:.2f} {Lmaxd[1]:.2f}  '
        dir_str = dir_str + f'{G1mind[1]:.2f} {G1maxd[1]:.2f}  '
        dir_str = dir_str + f'{P1mind[1]:.2f} {P1maxd[1]:.2f} ' + '\n'
        dir_str = dir_str + f'{Emind[2]:.2f} {Emaxd[2]:.2f}  '
        dir_str = dir_str + f'{Lmind[2]:.2f} {Lmaxd[2]:.2f}  '
        dir_str = dir_str + f'{G1mind[2]:.2f} {G1maxd[2]:.2f}  '
        dir_str = dir_str + f'{P1mind[2]:.2f} {P1maxd[2]:.2f} ' + '\n'
        dir_str = dir_str + 'x  x         x x        '
        dir_str = dir_str + f'{G2mind[0]:.2f} {G2maxd[0]:.2f}  '
        dir_str = dir_str + f'{P2mind[0]:.2f} {P2maxd[0]:.2f} ' + '\n'
        dir_str = dir_str + 'x  x         x x        '
        dir_str = dir_str + f'{G2mind[1]:.2f} {G2maxd[1]:.2f}  '
        dir_str = dir_str + f'{P2mind[1]:.2f} {P2maxd[1]:.2f} ' + '\n'
        dir_str = dir_str + 'x  x         x x        '
        dir_str = dir_str + f'{G2mind[2]:.2f} {G2maxd[2]:.2f}  '
        dir_str = dir_str + f'{P2mind[2]:.2f} {P2maxd[2]:.2f} ' + '\n'

    Cij_str = Cij_str + '\n'
    Eigenvalues_str = Eigenvalues_str + '\n'
    Av_str = Av_str + '\n'
    var_str = var_str + '\n'
    dir_str = dir_str + '\n'

    print(Cij_str)
    print(Eigenvalues_str)
    print(Av_str)
    print(var_str)

    print(dir_str)


def display_markdown_tables_FSparams(ndeb, DHf, S298, FS_db_params):
    txt2print = ['' for _ in range(5)]
    txt2print[0] = '| structure | DHf (kJ/mol-atom) | S298 (J/mol-atom-K) | \n |-------- | ---- | ---- |'
    txt2print[
        1] = '| structure | C0 | C1 | C2 | C3 | C4 | C5 |  \n |-------- | ---- | ---- | ---- | ---- | ---- | ---- |'
    txt2print[2] = '| structure | a0 | a1 | a2 | a3 | \n |-------- | ---- | ---- | ---- | ---- |'
    txt2print[3] = '| structure | invK0 |invK1 | invK2 | invK3 | \n |-------- | ---- | ---- | ---- | ---- |'
    txt2print[4] = '| structure | Ksp0 | Ksp1 | \n |-------- | ---- | ---- |'

    for structure in ndeb.keys():
        line = f'| {structure} | {DHf[structure]: .6e} | {S298[structure]: .6e} |'
        txt2print[0] += '\n' + line
    for structure in ndeb.keys():
        line = '| ' + structure + ' | '
        for key, val in FS_db_params[structure].items():
            line = '| ' + structure + ' | ' + ' | '.join([f"{v: .6e}" for v in val]) + ' | '
            if key == 'Cp':
                txt2print[1] += '\n' + line
            elif key == 'a':
                txt2print[2] += '\n' + line
            elif key == '1/Ks':
                txt2print[3] += '\n' + line
            elif key == 'Ksp':
                txt2print[4] += '\n' + line
    txt2print[0] = 'Formation enthalpy and Entropy at 298.15 K for all structures:\n' + txt2print[0]
    txt2print[1] = 'Heat Capacity parameters:\n' + txt2print[1]
    txt2print[2] = 'Thermal Expansion parameters:\n' + txt2print[2]
    txt2print[3] = 'Inverse Adiabatic Bulk Modulus parameters:\n' + txt2print[3]
    txt2print[4] = 'Pressure derivative of the Bulk Modulus parameters:\n' + txt2print[4]

    from IPython.display import display, Markdown, Latex
    for txti in txt2print:
        display(Markdown(txti))


import re


def extract_kpoints_from_log(log_filepath, encut, kspacing):
    target_encut = float(encut)
    target_kspacing = float(kspacing)

    kp_paths = []
    kp_texts = []

    with open(log_filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current = None
    inside_kp = False

    def finalize_current(cur):
        nonlocal kp_paths, kp_texts
        if cur is None:
            return
        if (cur.get("encut") is None or
                cur.get("kspacing") is None or
                not cur.get("kp_lines")):
            return

        # Filter by ENCUT and KSPACING (with small tolerances)
        if (abs(cur["encut"] - target_encut) >= 1e-6 or
                abs(cur["kspacing"] - target_kspacing) >= 1e-8):
            return

        full_path = cur["full_path"]
        parts = full_path.split("/")

        # Try to locate "s_*/sx_*" pair (generic, no hard-coded "Ca-Sr")
        s_idx = None
        for i, p in enumerate(parts):
            if p.startswith("s_") and i + 1 < len(parts) and parts[i + 1].startswith("sx_"):
                s_idx = i
                break

        if s_idx is not None:
            kp_path = "/".join(parts[s_idx:s_idx + 2])  # "s_0/sx_1"
        else:
            # Fallback: everything between 2nd and last component
            if len(parts) > 3:
                kp_path = "/".join(parts[2:-1])
            else:
                kp_path = full_path

        kp_paths.append(kp_path)
        # Build KPOINTS text (comment + 0 + Monkhorst-Pack + grid + shift)
        kp_texts.append("\n".join(cur["kp_lines"]) + "\n")

    for line in lines:
        stripped = line.rstrip("\n")

        # Start of a new block: "0: calcs_conv_420_0.10/Ca-Sr/s_0/sx_0/relaxation"
        m = re.match(r"^\s*\d+:\s+(.*?/relaxation)\s*$", stripped)
        if m:
            finalize_current(current)

            full_path = m.group(1).strip()
            current = {
                "full_path": full_path,
                "encut": None,
                "kspacing": None,
                "kp_lines": [],
            }
            inside_kp = False
            continue

        if current is None:
            continue

        # ENCUT_eV: 420.0
        if "ENCUT_eV" in stripped:
            try:
                current["encut"] = float(stripped.split("ENCUT_eV:")[1])
            except Exception:
                pass

        # KSPACING: 0.1
        if "KSPACING:" in stripped:
            try:
                current["kspacing"] = float(stripped.split("KSPACING:")[1])
            except Exception:
                pass

        # Start of KPOINTS suggestion block
        if "Suggested explicit KPOINTS (from KSPACING & reciprocal lattice)" in stripped:
            current["kp_lines"] = []
            inside_kp = True
            continue

        # Lines inside the KPOINTS block:
        #   Automatic mesh inferred from irreducible k-points
        #   0
        #   Monkhorst-Pack
        #   Nx Ny Nz
        #   0 0 0
        if inside_kp:
            if stripped == "":
                inside_kp = False
            else:
                current["kp_lines"].append(stripped)
            continue

    # Final block
    finalize_current(current)

    return {"kp_path": kp_paths, "text": kp_texts}


def write_kpoints_to_files(kps_txt, base_dir):
    for kp_path, kp_text in zip(kps_txt["kp_path"], kps_txt["text"]):
        kp_path_destination = f"{base_dir}/{kp_path}/KPOINTS"

        with open(kp_path_destination, "w", encoding="utf-8") as f:
            f.write(kp_text)


def count_if(lst, value):
    return lst.count(value)

def get_elements_from_poscar(poscar_fpath):
    with open(poscar_fpath) as f:
        lines = f.readlines()
        elements = lines[5].split()
        return set(elements)

from pathlib import Path
import os
def copy_tree_structure_only(src, dst, max_rel_depth=3, collect_rel_depth=3):
    src = Path(src).expanduser().resolve()
    dst = Path(dst).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    lst_of_folders = []

    for root, dirs, _files in os.walk(src):
        root = Path(root)
        rel = root.relative_to(src)
        depth = len(rel.parts)

        if depth > max_rel_depth:
            dirs[:] = []  # prune deeper traversal
            continue

        (dst / rel).mkdir(parents=True, exist_ok=True)

        if depth == collect_rel_depth:
            lst_of_folders.append(rel)

    return lst_of_folders

import re

HEADER_RE = re.compile(r'^\s*\d+:\s+(.+?)\s*$')
SSX_RE = re.compile(r'/(s_\d+)/(sx_\d+)')

E0_RE = re.compile(r'^\s*E0_eV:\s*([-\d\.Ee+]+)\s*$')
TOTEN_RE = re.compile(r'^\s*TOTEN_eV:\s*([-\d\.Ee+]+)\s*$')
NIONS_RE = re.compile(r'^\s*NIONS:\s*([-\d\.Ee+]+)\s*$')


def _key_from_path(path: str) -> str:
    """Prefer a compact key like 's_0/sx_3' if present; else use full path."""
    m = SSX_RE.search(path)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return path


def parse_stage_metrics(
    log_path: str
):
    """
    Parse a single stage log and return:
      e0_by_key, toten_by_key, nions_by_key
    """
    e0_by_key: Dict[str, float] = {}
    toten_by_key: Dict[str, float] = {}
    nions_by_key: Dict[str, int] = {}
    current_path = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            mh = HEADER_RE.match(line)
            if mh:
                current_path = mh.group(1)
                continue

            if not current_path:
                continue

            key = _key_from_path(current_path)

            m = E0_RE.match(line)
            if m:
                try:
                    e0_by_key[key] = float(m.group(1))
                except ValueError:
                    pass
                continue

            m = TOTEN_RE.match(line)
            if m:
                try:
                    toten_by_key[key] = float(m.group(1))
                except ValueError:
                    pass
                continue

            m = NIONS_RE.match(line)
            if m:
                try:
                    nions_by_key[key] = int(float(m.group(1)))
                except ValueError:
                    pass
                continue

    return e0_by_key, toten_by_key, nions_by_key


def extract_relax_vs_static(
    relax_log: str,
    static_log: str
):
    """
    Returns:
      E0_dict[key]    = [E0_relax, E0_static]
      TOTEN_dict[key] = [TOTEN_relax, TOTEN_static]
      NIONS_dict[key] = [NIONS_relax, NIONS_static]
    """
    relax_e0, relax_toten, relax_nions = parse_stage_metrics(relax_log)
    static_e0, static_toten, static_nions = parse_stage_metrics(static_log)

    keys = sorted(
        set(relax_e0) | set(static_e0)
        | set(relax_toten) | set(static_toten)
        | set(relax_nions) | set(static_nions)
    )

    e0_pair = {k: [relax_e0.get(k), static_e0.get(k)] for k in keys}
    toten_pair = {k: [relax_toten.get(k), static_toten.get(k)] for k in keys}
    nions_pair = {k: [relax_nions.get(k), static_nions.get(k)] for k in keys}

    return e0_pair, toten_pair, nions_pair

def print_sumtest(dc_conv_test):
    print('Total number of calculations:', len(dc_conv_test))
    dc_jobs_ok = dc_conv_test.filter_by_value('overall_status', 'ok')
    print('- jobs completed successfully:', len(dc_jobs_ok))
    dc_jobs_err = dc_conv_test.filter_by_value('overall_status', 'error')
    print('- jobs with errors:', len(dc_jobs_err))
    types_of_errors = set(dc_jobs_err['error_type'])
    for err_type in types_of_errors:
        dc_err_type = dc_jobs_err.filter_by_value('error_type', err_type)
        print(f'  - {err_type}: {len(dc_err_type)}')
    print('- By KSPACING:')
    for kspacing in sorted(set(dc_conv_test['kspacing'])):
        dc_kspacing = dc_jobs_err.filter_by_value('kspacing', kspacing)
        print(f'  - {kspacing}: {len(dc_kspacing)}')
    print('- By ENCUT:')
    for encut in sorted(set(dc_conv_test['encut'])):
        dc_encut = dc_jobs_err.filter_by_value('encut', encut)
        print(f'  - {encut}: {len(dc_encut)}')