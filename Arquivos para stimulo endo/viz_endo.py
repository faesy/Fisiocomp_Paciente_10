#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador interativo: .alg (pontos) + .msh (malha)
- Lê 2 arquivos: .alg (CSV simples), .msh (Gmsh)
- .msh é escalado por 1000 ao carregar
- Mostra a nuvem do .alg e a malha do .msh (superfície quando possível)
- Translada APENAS o .msh via ΔX, ΔY, ΔZ (digitados), multiplicados por 1000
- Botão "Encaixar (ICP)" para alinhar o .msh ao .alg (Open3D)
"""

import sys
from pathlib import Path
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QGridLayout, QLabel,
    QDoubleSpinBox, QPushButton, QMessageBox
)

# =========================
# ======== CONFIG =========
# =========================
ALG_PATH = "../Coração .Alg/Patient_10.alg"
MSH_PATH = "../Arquivos para stimulo endo/Patient_10.msh" 

SUBSAMPLE = 10
MSH_SCALE = 1000.0  # <<<<< ULTIPLICADOR DAS COORDENADAS DO MSH

# translação inicial (aplicada uma única vez ao .msh após o carregamento)
INIT_TX, INIT_TY, INIT_TZ = 0.0, 0.0, 0.0

# multiplicador aplicado aos valores digitados nas caixas de texto
MOVE_MULTIPLIER = 1000.0

POINT_SIZE_ALG = 3.0
POINT_SIZE_MSH = 3.0
WINDOW_SIZE = (1200, 900)

COLOR_ALG = "black"
COLOR_MSH = (1.0, 0.2, 0.2)
OPACITY_ALG = 0.95
OPACITY_MSH = 0.9
# =========================


def load_alg_points(path, subsample=10):
    """Lê arquivo .alg (CSV simples: x,y,z)."""
    coords = np.genfromtxt(path, delimiter=",", usecols=(0, 1, 2))
    if coords.ndim == 1:
        coords = coords[None, :]
    if subsample > 1:
        coords = coords[::subsample]
    return coords


def load_msh_as_polydata(path: str, scale=1.0) -> pv.PolyData:
    """
    Tenta carregar .msh como PolyData:
    1) pv.read + extract_surface
    2) (fallback) meshio -> PolyData com 'triangle'
    3) (fallback final) nuvem de pontos
    """
    try:
        mesh = pv.read(path)
        if isinstance(mesh, pv.MultiBlock):
            mesh = mesh.combine()

        try:
            surf = mesh.extract_surface()
            if surf.n_points > 0:
                mesh = surf
        except Exception:
            pass

        if not isinstance(mesh, pv.PolyData):
            mesh = pv.wrap(mesh)

        if scale != 1.0:
            mesh.points *= float(scale)

        return mesh
    except Exception:
        # fallback com meshio
        try:
            import meshio
        except Exception:
            raise RuntimeError(
                "Falha ao ler .msh com PyVista e 'meshio' não está instalado.\n"
                "Instale com: pip install meshio"
            )
        m = meshio.read(path)
        pts = m.points.astype(float) * float(scale)

        tri = None
        if hasattr(m, "cells_dict"):
            tri = m.cells_dict.get("triangle", None)
        else:
            for c in m.cells:
                if c.type in ("triangle", "tri"):
                    tri = c.data
                    break

        if tri is not None and len(tri) > 0:
            faces = np.hstack([np.full((tri.shape[0], 1), 3, dtype=np.int64), tri]).ravel()
            return pv.PolyData(pts, faces)
        else:
            return pv.PolyData(pts)


def _extent(points: np.ndarray) -> float:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return float(np.max(maxs - mins))

def _pvpoly_to_o3d_pc(points: np.ndarray):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    return pc

def _centroid(pts: np.ndarray) -> np.ndarray:
    return np.mean(pts, axis=0)

def run_icp_align(alg_pts: np.ndarray, msh_pts: np.ndarray) -> np.ndarray:
    """
    Tenta alinhar MSH->ALG.
    1) Pré-alinha por centróides.
    2) ICP point-to-plane; se fitness baixo, ICP point-to-point.
    Retorna T 4x4 homogênea.
    """
    import open3d as o3d

    # --- pré-alinhamento por centróides (apenas translação) ---
    c_alg = _centroid(alg_pts)
    c_msh = _centroid(msh_pts)
    T_pre = np.eye(4)
    T_pre[:3, 3] = (c_alg - c_msh)

    msh_pre = (msh_pts + (c_alg - c_msh))

    # --- downsample adaptativo e seguros ---
    ext_alg = _extent(alg_pts)
    ext_msh = _extent(msh_pre)
    extent = max(ext_alg, ext_msh)
    # voxel não pode ser grande demais nem zerar:
    voxel = max(extent * 0.005, 1e-3)   # 0.5% do bbox, mínimo 1e-3
    thres = max(extent * 0.02, 5 * voxel)  # 2% do bbox, mínimo 5 voxels

    tgt = _pvpoly_to_o3d_pc(alg_pts).voxel_down_sample(voxel)
    src = _pvpoly_to_o3d_pc(msh_pre).voxel_down_sample(voxel)

    # se sobrar muito pouco ponto, cai fora cedo
    print(f"[ICP] pts alg={len(alg_pts)}→{len(tgt.points)}  "
          f"msh={len(msh_pts)}→{len(src.points)}  voxel={voxel:.6g}  thr={thres:.6g}")

    if len(tgt.points) < 50 or len(src.points) < 50:
        print("[ICP] Muito poucos pontos após downsample. "
              "Aumente SUBSAMPLE do .alg (menor amostragem) ou reduza MSH_SCALE.")
        # retorna só o pré-alinhamento de centróides
        return T_pre

    # estima normais para point-to-plane
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3*voxel, max_nn=50))
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3*voxel, max_nn=50))

    init = np.eye(4)
    best_T = np.eye(4)
    best_fit = -1.0
    best_rmse = np.inf

    # 1) point-to-plane
    reg_plane = o3d.pipelines.registration.registration_icp(
        source=src, target=tgt,
        max_correspondence_distance=thres,
        init=init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    print(f"[ICP plane] fitness={reg_plane.fitness:.4f} rmse={reg_plane.inlier_rmse:.6g}")

    if reg_plane.fitness > best_fit or (reg_plane.fitness == best_fit and reg_plane.inlier_rmse < best_rmse):
        best_fit = reg_plane.fitness
        best_rmse = reg_plane.inlier_rmse
        best_T = reg_plane.transformation

    # 2) se fitness baixo, tenta point-to-point
    if best_fit < 0.1:
        reg_point = o3d.pipelines.registration.registration_icp(
            source=src, target=tgt,
            max_correspondence_distance=thres,
            init=init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
        )
        print(f"[ICP point] fitness={reg_point.fitness:.4f} rmse={reg_point.inlier_rmse:.6g}")
        if reg_point.fitness > best_fit or (reg_point.fitness == best_fit and reg_point.inlier_rmse < best_rmse):
            best_fit = reg_point.fitness
            best_rmse = reg_point.inlier_rmse
            best_T = reg_point.transformation

    # composição: T_total = T_pre * best_T
    T_total = np.eye(4)
    T_total[:] = T_pre @ best_T
    print("[ICP] T_total =\n", T_total)
    return T_total

def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return (points @ R.T) + t



def main():
    base = Path(__file__).parent
    alg_path = (base / ALG_PATH).resolve()
    msh_path = (base / MSH_PATH).resolve()

    # --- .alg (nuvem)
    alg_pts = load_alg_points(str(alg_path), subsample=SUBSAMPLE)
    alg_cloud = pv.PolyData(alg_pts)

    # --- .msh (superfície quando possível; senão, pontos) + escala x1000
    msh_poly = load_msh_as_polydata(str(msh_path), scale=MSH_SCALE).copy()

    # translação inicial no .msh (opcional)
    if any(abs(a) > 1e-12 for a in (INIT_TX, INIT_TY, INIT_TZ)):
        msh_poly.points += np.array([INIT_TX, INIT_TY, INIT_TZ], dtype=float)

    # guarda base para translação interativa
    msh_base = msh_poly.points.copy()

    # -------- PyQt5 interface --------
    app = QApplication.instance() or QApplication(sys.argv)
    win = QWidget()
    win.setWindowTitle(".ALG + .MSH — Translação do MSH ×1000 / ICP")
    win.resize(*WINDOW_SIZE)

    layout = QGridLayout(win)
    plotter = QtInteractor(win)
    layout.addWidget(plotter, 0, 0, 1, 4)

    # labels
    layout.addWidget(QLabel("ΔX"), 1, 0)
    layout.addWidget(QLabel("ΔY"), 2, 0)
    layout.addWidget(QLabel("ΔZ"), 3, 0)

    spin_dx = QDoubleSpinBox()
    spin_dy = QDoubleSpinBox()
    spin_dz = QDoubleSpinBox()
    for sp in (spin_dx, spin_dy, spin_dz):
        sp.setRange(-1e6, 1e6)
        sp.setDecimals(4)
        sp.setSingleStep(0.1)
    layout.addWidget(spin_dx, 1, 1)
    layout.addWidget(spin_dy, 2, 1)
    layout.addWidget(spin_dz, 3, 1)

    apply_btn = QPushButton("Aplicar")
    reset_btn = QPushButton("Reset")
    icp_btn   = QPushButton("Encaixar (ICP)")
    layout.addWidget(apply_btn, 4, 1)
    layout.addWidget(reset_btn, 4, 2)
    layout.addWidget(icp_btn,   4, 3)

    # render inicial
    pv.set_plot_theme("document")
    plotter.add_axes()
    plotter.show_grid()
    plotter.enable_eye_dome_lighting()

    plotter.add_mesh(
        alg_cloud,
        render_points_as_spheres=True,
        point_size=POINT_SIZE_ALG,
        color=COLOR_ALG,
        opacity=OPACITY_ALG,
        name="ALG",
    )

    render_as_points = isinstance(msh_poly, pv.PolyData) and (msh_poly.n_faces == 0)
    plotter.add_mesh(
        msh_poly,
        render_points_as_spheres=render_as_points,
        point_size=(POINT_SIZE_MSH if render_as_points else 1.0),
        color=COLOR_MSH,
        opacity=OPACITY_MSH,
        name="MSH",
    )

    plotter.camera_position = "yz"

    def apply_translation():
        dx = spin_dx.value() * MOVE_MULTIPLIER
        dy = spin_dy.value() * MOVE_MULTIPLIER
        dz = spin_dz.value() * MOVE_MULTIPLIER
        shift = np.array([dx, dy, dz], dtype=float)
        msh_poly.points[:] = msh_base + shift
        plotter.update()
        print(f"[APPLY] Δ=({dx:.3f}, {dy:.3f}, {dz:.3f}) (×{MOVE_MULTIPLIER})")

    def reset_translation():
        for sp in (spin_dx, spin_dy, spin_dz):
            sp.setValue(0.0)
        # volta ao estado base atual (pode ser pós-ICP)
        msh_poly.points[:] = msh_base
        plotter.update()
        print("[RESET] ΔX/ΔY/ΔZ = 0 (voltou para a base atual)")

    def run_icp():
        try:
            import open3d  # checa dependência
        except Exception:
            QMessageBox.critical(
                win, "Open3D não instalado",
                "Para usar o ICP, instale o Open3D:\n\npip install open3d"
            )
            return

        try:
            T = run_icp_align(alg_pts, msh_poly.points)
        except Exception as e:
            QMessageBox.critical(win, "ICP falhou", f"Erro ao rodar ICP:\n{e}")
            return

        new_pts = apply_transform(msh_poly.points, T)
        msh_poly.points[:] = new_pts
        plotter.update()

        nonlocal msh_base
        msh_base = new_pts.copy()
        for sp in (spin_dx, spin_dy, spin_dz):
            sp.setValue(0.0)

        print("[ICP] Transformação aplicada e base atualizada.")

            # ---- salva deslocamento (Tx, Ty, Tz) em TXT ----
        t = T[:3, 3]
        out_path = (base / "msh_icp_shift.txt").resolve()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"ΔX = {t[0]:.6f}\nΔY = {t[1]:.6f}\nΔZ = {t[2]:.6f}\n")
        print(f"[SAVE] deslocamento ICP salvo em: {out_path}")



    apply_btn.clicked.connect(apply_translation)
    reset_btn.clicked.connect(reset_translation)
    icp_btn.clicked.connect(run_icp)

    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
