#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pyvista as pv

# =========================
# ======== CONFIG =========
# =========================
BASE_DIR = Path(__file__).parent  # pasta do script

SHIFT_TXT = BASE_DIR / "msh_icp_shift.txt"

LV_STL_IN = BASE_DIR / "Patient_1-LVEndo.stl"
RV_STL_IN = BASE_DIR / "Patient_1-RVEndo.stl"

LV_VTK_OUT = BASE_DIR / "Patient_1-LVEndo_shifted.vtk"
RV_VTK_OUT = BASE_DIR / "Patient_1-RVEndo_shifted.vtk"

# escala aplicada aos STLs antes da translação
STL_SCALE = 1000.0
# =========================


def parse_shift_txt(path: Path) -> np.ndarray:
    """
    Lê o arquivo de deslocamento e retorna vetor np.array([dx, dy, dz]).
    Aceita linhas como: 'ΔX = 123.456' ou 'DX=123,456' (vírgula/ponto).
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de deslocamento não encontrado: {path}")

    # valores padrão caso alguma linha não exista
    dx = dy = dz = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip().lower().replace(" ", "")
            s = s.replace("δ", "d")  # caso venha com delta grego
            s = s.replace(",", ".")  # aceita vírgula decimal
            if s.startswith("dx=") or s.startswith("Δx=".lower()):
                dx = float(s.split("=", 1)[1])
            elif s.startswith("dy=") or s.startswith("Δy=".lower()):
                dy = float(s.split("=", 1)[1])
            elif s.startswith("dz=") or s.startswith("Δz=".lower()):
                dz = float(s.split("=", 1)[1])

    # fallback: tentar ler números em ordem se labels não estiverem presentes
    if dx is None or dy is None or dz is None:
        vals = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                # pega primeira coluna numérica encontrada
                tokens = ln.replace(",", ".").replace("=", " ").split()
                for tok in tokens:
                    try:
                        vals.append(float(tok))
                        break
                    except Exception:
                        continue
        if len(vals) >= 3:
            dx = vals[0] if dx is None else dx
            dy = vals[1] if dy is None else dy
            dz = vals[2] if dz is None else dz

    if dx is None or dy is None or dz is None:
        raise ValueError(f"Não foi possível interpretar ΔX/ΔY/ΔZ em: {path}")

    return np.array([dx, dy, dz], dtype=float)


def process_one_stl(stl_in: Path, vtk_out: Path, scale: float, shift: np.ndarray):
    """
    Lê STL, aplica escala e translação, salva como .vtk.
    """
    if not stl_in.exists():
        raise FileNotFoundError(f"STL não encontrado: {stl_in}")

    mesh = pv.read(str(stl_in)).extract_surface()  # garante superfície
    pts = mesh.points.astype(np.float64, copy=True)

    # 1) escala
    if scale != 1.0:
        pts *= float(scale)

    # 2) translação
    pts += shift

    mesh.points[:] = pts

    # 3) salva .vtk (legacy VTK)
    mesh.save(str(vtk_out))
    print(f"[OK] Salvo: {vtk_out}  (pontos: {mesh.n_points})")


def main():
    shift = parse_shift_txt(SHIFT_TXT)
    print(f"[SHIFT] Δ = ({shift[0]:.6f}, {shift[1]:.6f}, {shift[2]:.6f})")

    process_one_stl(LV_STL_IN, LV_VTK_OUT, STL_SCALE, shift)
    process_one_stl(RV_STL_IN, RV_VTK_OUT, STL_SCALE, shift)

    print("[DONE] Conversão e deslocamento concluídos.")


if __name__ == "__main__":
    main()
