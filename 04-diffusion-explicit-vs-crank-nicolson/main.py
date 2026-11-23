import os
import argparse
from pathlib import Path
import numpy as np

from plotter import *
from metodos_numericos import *


def _parse_list(s: str, cast=float):
    """
    Parsea listas tipo "1,2,3" -> [1,2,3]
    """
    if s is None:
        return None
    # Permite comas o espacios
    parts = []
    for chunk in s.replace(",", " ").split():
        if chunk.strip():
            parts.append(cast(chunk))
    return parts


def _parse_Cs(s: str):
    """
    Parsea lista de Cs
    """
    if s is None:
        return None
    tokens = s.replace(",", " ").split()
    out = []
    for tok in tokens:
        t = tok.strip().lower()
        out.append(float(tok))
    return out


def _ensure_outdir(output_path: str) -> str:
    p = Path(output_path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    # Devolvemos con separador final para mantener  concatenaciones de strings
    return str(p) + os.sep


def run_calculo(args):
    # Defaults por modo si no vienen por CLI
    T_final = args.T_final if args.T_final is not None else 0.5

    x, t = crear_malla(L=args.L, N_div_x=args.N_div_x, T_final=T_final, N_div_t=args.N_div_t)
    dx = x[1] - x[0]

    mask_pos = (x >= 0.0) if args.solo_mitad else np.ones_like(x, dtype=bool)

    # Soluciones
    n_num = diffusion_numeric(x, t, D=args.D, C=args.C)
    n_cn  = diffusion_cn(x, t, D=args.D, C=args.C)
    n_exa = diffusion_exact(x, t, D=args.D, C=args.C, N_modes=args.N_modes_exact)

    # Errores (en dominio elegido)
    error = np.abs(n_num - n_exa)[:, mask_pos]
    max_err_t = error.max(axis=1)
    l2_err_t = np.sqrt((error**2).sum(axis=1) * dx)

    error_cn = np.abs(n_cn - n_exa)[:, mask_pos]
    max_err_t_cn = error_cn.max(axis=1)
    l2_err_t_cn = np.sqrt((error_cn**2).sum(axis=1) * dx)

    # Gráficas a tiempos dados
    tiempos_a_pintar = _parse_list(args.tiempos_a_pintar, float)
    plot_comparacion_soluciones(
        x[mask_pos], t, n_num[:, mask_pos], n_exa[:, mask_pos],
        tiempos_a_pintar, args.output_path, "progresivo"
    )
    plot_comparacion_soluciones(
        x[mask_pos], t, n_cn[:, mask_pos], n_exa[:, mask_pos],
        tiempos_a_pintar, args.output_path, "CN"
    )

    # CSVs
    tiempos_csv = _parse_list(args.tiempos_csv, float)
    idx_times = np.array([int(np.argmin(np.abs(t - tt))) for tt in tiempos_csv], dtype=int)

    # CSV 1: errores máximos y nodo correspondiente (en el dominio elegido)
    rows_err = []
    for k_idx in idx_times:
        tk = t[k_idx]

        err_exp_k = np.abs(n_num[k_idx, mask_pos] - n_exa[k_idx, mask_pos])
        max_err_exp = float(np.max(err_exp_k))
        idx_max_exp_local = int(np.argmax(err_exp_k))  # índice dentro de mask_pos

        err_cn_k = np.abs(n_cn[k_idx, mask_pos] - n_exa[k_idx, mask_pos])
        max_err_cn = float(np.max(err_cn_k))
        idx_max_cn_local = int(np.argmax(err_cn_k))

        rows_err.append([tk, max_err_exp, idx_max_exp_local, max_err_cn, idx_max_cn_local])

    rows_err = np.array(rows_err, dtype=float)
    csv_path_err = args.output_path + "errores_maximos_por_tiempo.csv"
    header_err = "t,max_err_exp,idx_max_exp_local,max_err_cn,idx_max_cn_local"
    np.savetxt(csv_path_err, rows_err, delimiter=",", header=header_err, comments="")
    print(f"CSV de errores máximos guardado en: {csv_path_err}")

    # CSV 2: soluciones explícitas
    rows_exp = []
    for k_idx in idx_times:
        tk = t[k_idx]
        fila = np.concatenate(([tk], n_num[k_idx, mask_pos]))
        rows_exp.append(fila)
    rows_exp = np.vstack(rows_exp)
    csv_path_exp = args.output_path + "sol_exp_t_x.csv"
    header_exp = "t," + ",".join([f"n_exp_x{i}" for i in range(mask_pos.sum())])
    np.savetxt(csv_path_exp, rows_exp, delimiter=",", header=header_exp, comments="")

    # CSV 3: soluciones CN
    rows_cn = []
    for k_idx in idx_times:
        tk = t[k_idx]
        fila = np.concatenate(([tk], n_cn[k_idx, mask_pos]))
        rows_cn.append(fila)
    rows_cn = np.vstack(rows_cn)
    csv_path_cn = args.output_path + "sol_cn_t_x.csv"
    header_cn = "t," + ",".join([f"n_cn_x{i}" for i in range(mask_pos.sum())])
    np.savetxt(csv_path_cn, rows_cn, delimiter=",", header=header_cn, comments="")

    # Densidad promedio
    n_prom_num = densidad_promedio(x, n_cn, solo_mitad=args.solo_mitad)
    plot_densidad_promedio(t, n_prom_num, args.output_path)


def run_convergencia(args):
    T_final = args.T_final if args.T_final is not None else 0.1

    Nxs = _parse_list(args.Nxs, int)
    h_list_prog, errL2_prog = [], []
    h_list_cn,   errL2_cn   = [], []

    for N_div_x in Nxs:
        h, eL2, _ = calcular_error_espacio(
            N_div_x=N_div_x,
            N_div_t=args.N_div_t,
            T_final=T_final,
            D=args.D,
            C=args.C,
            L=args.L,
            N_modes=args.N_modes
        )
        h_list_prog.append(h)
        errL2_prog.append(eL2)

        h_cn, eL2_cn, _ = calcular_error_espacio_cn(
            N_div_x=N_div_x,
            N_div_t=args.N_div_t,
            T_final=T_final,
            D=args.D,
            C=args.C,
            L=args.L,
            N_modes=args.N_modes
        )
        h_list_cn.append(h_cn)
        errL2_cn.append(eL2_cn)

        print(f"Nx={N_div_x:3d}  h={h:.3e}  PROG: L2={eL2:.3e}   CN: L2={eL2_cn:.3e}")

    h_arr_prog = np.array(h_list_prog)
    h_arr_cn   = np.array(h_list_cn)
    errL2_prog = np.array(errL2_prog)
    errL2_cn   = np.array(errL2_cn)

    p_L2_prog, _ = np.polyfit(np.log(h_arr_prog), np.log(errL2_prog), 1)
    p_L2_cn,   _ = np.polyfit(np.log(h_arr_cn),   np.log(errL2_cn),   1)

    print(f"Orden espacial PROG (L2):  {p_L2_prog:.2f}")
    print(f"Orden espacial  C-N (L2):  {p_L2_cn:.2f}")

    plot_convergencia_espacial_dos_metodos(h_arr_prog, errL2_prog, h_arr_cn, errL2_cn)


def run_convergencia_tiempo(args):
    T_final = args.T_final if args.T_final is not None else 0.1
    Nts = _parse_list(args.Nts, int)

    dt_prog_list, errL2_prog_list = [], []
    dt_cn_list,   errL2_cn_list   = [], []

    for N_div_t in Nts:
        dt_prog, eL2_prog = calcular_error_tiempo_prog(
            N_div_x=args.N_div_x,
            N_div_t=N_div_t,
            T_final=T_final,
            D=args.D,
            C=args.C,
            L=args.L,
            N_modes=args.N_modes
        )
        dt_prog_list.append(dt_prog)
        errL2_prog_list.append(eL2_prog)

        dt_cn, eL2_cn = calcular_error_tiempo_cn(
            N_div_x=args.N_div_x,
            N_div_t=N_div_t,
            T_final=T_final,
            D=args.D,
            C=args.C,
            L=args.L,
            N_modes=args.N_modes
        )
        dt_cn_list.append(dt_cn)
        errL2_cn_list.append(eL2_cn)

        print(f"Nt={N_div_t:5d}  dt={dt_prog:.3e}  PROG: L2={eL2_prog:.3e}   CN: L2={eL2_cn:.3e}")

    dt_prog_arr = np.array(dt_prog_list)
    errL2_prog_arr = np.array(errL2_prog_list)
    dt_cn_arr = np.array(dt_cn_list)
    errL2_cn_arr = np.array(errL2_cn_list)

    p_L2_prog, _ = np.polyfit(np.log(dt_prog_arr), np.log(errL2_prog_arr), 1)
    p_L2_cn,   _ = np.polyfit(np.log(dt_cn_arr),   np.log(errL2_cn_arr),   1)

    print(f"Orden temporal PROG (L2): {p_L2_prog:.2f}")
    print(f"Orden temporal  C-N (L2): {p_L2_cn:.2f}")

    plot_convergencia_tiempo_dos_metodos(dt_prog_arr, errL2_prog_arr, dt_cn_arr, errL2_cn_arr)


def run_general_C(args):
    T_final = args.T_final if args.T_final is not None else 1.0

    x, t = crear_malla(L=args.L, N_div_x=args.N_div_x, T_final=T_final, N_div_t=args.N_div_t)
    Cs = _parse_Cs(args.Cs)

    n_prom_dict = {}
    for C_val in Cs:
        n_cn = diffusion_cn(x, t, D=args.D, C=C_val)
        n_prom = densidad_promedio(x, n_cn, solo_mitad=args.solo_mitad)
        n_prom_dict[C_val] = n_prom

    plot_densidad_promedio_multi(t, n_prom_dict, args.output_path)


def build_parser():
    p = argparse.ArgumentParser(
        description="Script parametrizable para difusión: calculo / convergencia / convergencia_tiempo / general_C"
    )

    # Comunes
    p.add_argument("--mode", required=True, choices=["calculo", "convergencia", "convergencia_tiempo", "general_C"])
    p.add_argument("--L", type=float, default=1.0)
    p.add_argument("--D", type=float, default=1.0)
    p.add_argument("--C", type=float, default=0.0, help="Usado en todos los modos excepto general_C (que usa --Cs).")
    p.add_argument("--T_final", type=float, default=None, help="Si no se pasa, se usa el default del modo.")
    p.add_argument("--output_path", type=str, default=".", help="Directorio donde guardar salidas.")
    p.add_argument("--solo_mitad", action=argparse.BooleanOptionalAction, default=True,
                   help="Usar solo x>=0 (simetría). Default: True. Usa --no-solo_mitad para dominio completo.")

    # Parámetros por modo (se declaran todos; solo se usarán según mode)
    # calculo
    p.add_argument("--N_div_x", type=int, default=20, help="(calculo/general_C) divisiones espaciales")
    p.add_argument("--N_div_t", type=int, default=5000, help="(calculo/convergencia/general_C) divisiones temporales")
    p.add_argument("--N_modes_exact", type=int, default=400, help="(calculo) modos para solución exacta")
    p.add_argument("--tiempos_a_pintar", type=str,
                   default="0.005,0.01,0.02,0.05,0.1,0.5",
                   help="(calculo) lista de tiempos para gráficas. Ej: '0.01,0.02,0.05'")
    p.add_argument("--tiempos_csv", type=str,
                   default="0.001,0.005,0.01,0.015,0.02,0.03,0.05,0.075,0.1,0.2,0.5",
                   help="(calculo) lista de tiempos para CSV")

    # convergencia espacial
    p.add_argument("--Nxs", type=str, default="5,10,20,40,80,160",
                   help="(convergencia) lista de Nx. Ej: '10,20,40,80'")
    p.add_argument("--N_modes", type=int, default=800, help="(convergencia/convergencia_tiempo) modos exacta")

    # convergencia temporal
    p.add_argument("--Nts", type=str, default="1280,2560,5120,10240",
                   help="(convergencia_tiempo) lista de Nt. Ej: '1000,2000,4000'")

    # general_C
    p.add_argument("--Cs", type=str,
                   default="0,2,5,8,9.5,pi2,10.5,12",
                   help="(general_C) lista de C. Permite 'pi2' para pi^2.")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Normaliza/crea output dir
    args.output_path = _ensure_outdir(args.output_path)

    if args.mode == "calculo":
        run_calculo(args)
    elif args.mode == "convergencia":
        run_convergencia(args)
    elif args.mode == "convergencia_tiempo":
        run_convergencia_tiempo(args)
    elif args.mode == "general_C":
        run_general_C(args)
    else:
        raise ValueError(f"Modo no soportado: {args.mode}")


if __name__ == "__main__":
    main()
