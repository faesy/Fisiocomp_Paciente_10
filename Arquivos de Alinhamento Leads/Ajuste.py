import sys
import json
import numpy as np
import numpy.linalg as npl
import pandas as pd
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QDoubleSpinBox,
    QPushButton, QGridLayout, QCheckBox
)
from scipy.spatial.transform import Rotation as R


def make_T_from_R_t_pivot(R_mat: np.ndarray, t_vec: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    """
    T = T_t · T_p · T_R · T_-p
    (rotação em torno de 'pivot' e translação global aplicada por último)
    """
    T_p  = np.eye(4);  T_p[:3, 3]  = pivot
    T_np = np.eye(4);  T_np[:3, 3] = -pivot
    T_R  = np.eye(4);  T_R[:3, :3] = R_mat
    T_t  = np.eye(4);  T_t[:3, 3]  = t_vec
    return T_t @ T_p @ T_R @ T_np


class JanelaControle(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Alinhamento de Malha VTP ao .ALG (fixo)")

        layout = QVBoxLayout()
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        # --- Carrega .alg (FIXO) ---
        df = pd.read_csv("Paciente 1/Coração .Alg/Patient_1.alg", header=None)
        self.points_alg = df[[0, 1, 2]].values.astype(float) / 1000.0
        self.pontos_alg = pv.PolyData(self.points_alg)
        valores = df.iloc[:, 8:17].values.astype(float)
        for i in range(valores.shape[1]):
            self.pontos_alg[f"campo_{8+i}"] = valores[:, i]

        # --- Carrega malha do coração (MÓVEL) ---
        self.malha_vtu_original = pv.read("Paciente 1/Segmentações/Segmentação Original/coração Abhi.vtp")
        self.malha_vtu = self.malha_vtu_original.copy()

        # Actor do coração (iluminação ligada; parâmetros compatíveis)
        self.actor_vtu = self.plotter.add_mesh(
            self.malha_vtu,
            scalars=None,
            color="#9aa0a6",       # cinza neutro (evita branco chapado)
            opacity=1.0,
            show_edges=False,
            smooth_shading=True,
            lighting=True,
            ambient=0.15,
            diffuse=0.9,
            specular=0.05,
            specular_power=10,
        )

        # Normais iniciais (uma vez). Depois só faremos flip quando necessário.
        try:
            if 'Normals' not in self.malha_vtu.point_data:
                self.malha_vtu.compute_normals(
                    inplace=True,
                    auto_orient_normals=True,
                    split_vertices=True,
                    feature_angle=50.0
                )
        except Exception:
            pass

        # --- Plot do .alg (fixo) ---
        self.plotter.add_mesh(
            self.pontos_alg,
            scalars="campo_8",
            render_points_as_spheres=True,
            point_size=5,
            color='red',
            lighting=False
        )

        # --- Estado (absoluto, sem acúmulo) ---
        self.offset = np.zeros(3, dtype=float)    # (tx, ty, tz)
        self.rotacao = np.zeros(3, dtype=float)   # (rx, ry, rz) graus
        self.mirror = np.array([1.0, 1.0, 1.0], dtype=float)  # espelhos X,Y,Z (±1)
        self.last_mirror_sign = 1.0  # produto anterior (+1: par; -1: ímpar)

        # Geometria e pivot originais
        self.P0 = self.malha_vtu_original.points.copy()
        self.c0 = np.array(self.malha_vtu_original.center, dtype=float)  # pivot fixo (centro do coração original)

        # --- UI: spinboxes ---
        grid = QGridLayout()
        self.spins = {}
        for i, eixo in enumerate("xyz"):
            # Translação
            grid.addWidget(QLabel(f"Transl. {eixo.upper()}"), i, 0)
            spin_t = QDoubleSpinBox()
            spin_t.setRange(-1000.0, 1000.0)
            spin_t.setDecimals(3)
            spin_t.valueChanged.connect(lambda v, ax=i: self.atualizar_translacao(ax, v))
            grid.addWidget(spin_t, i, 1)
            self.spins[f"t{eixo}"] = spin_t

            # Rotação
            grid.addWidget(QLabel(f"Rotac. {eixo.upper()} (°)"), i, 2)
            spin_r = QDoubleSpinBox()
            spin_r.setRange(-180.0, 180.0)
            spin_r.setDecimals(2)
            spin_r.valueChanged.connect(lambda v, ax=i: self.atualizar_rotacao(ax, v))
            grid.addWidget(spin_r, i, 3)
            self.spins[f"r{eixo}"] = spin_r

        # --- UI: espelhos (X/Y/Z) ---
        row_m = 3
        grid.addWidget(QLabel("Espelhar:"), row_m, 0)
        self.chk_mx = QCheckBox("X")
        self.chk_my = QCheckBox("Y")
        self.chk_mz = QCheckBox("Z")
        self.chk_mx.stateChanged.connect(lambda _: self.toggle_mirror(0))
        self.chk_my.stateChanged.connect(lambda _: self.toggle_mirror(1))
        self.chk_mz.stateChanged.connect(lambda _: self.toggle_mirror(2))
        grid.addWidget(self.chk_mx, row_m, 1)
        grid.addWidget(self.chk_my, row_m, 2)
        grid.addWidget(self.chk_mz, row_m, 3)

        layout.addLayout(grid)

        # Botões utilitários (opcionais)
        btn_reset_pose = QPushButton("Resetar Pose (R, t)")
        btn_reset_pose.clicked.connect(self.reset_pose)
        layout.addWidget(btn_reset_pose)

        btn_reset_mirror = QPushButton("Resetar Espelhos")
        btn_reset_mirror.clicked.connect(self.reset_mirror)
        layout.addWidget(btn_reset_mirror)

        # --- Botão salvar ---
        btn_salvar = QPushButton("Salvar Eletrodos e Torso")
        btn_salvar.clicked.connect(self.salvar_transformados)
        layout.addWidget(btn_salvar)

        self.setLayout(layout)

        # --- Render inicial ---
        self.plotter.show_axes()
        self.plotter.show_grid()
        self.plotter.reset_camera()

    # --- Callbacks ---
    def atualizar_translacao(self, eixo, novo_valor):
        self.offset[eixo] = float(novo_valor)
        self.aplicar_transformacao()

    def atualizar_rotacao(self, eixo, novo_valor):
        self.rotacao[eixo] = float(novo_valor)
        self.aplicar_transformacao()

    def toggle_mirror(self, axis_idx: int):
        # checkbox marcado => -1 (espelho); desmarcado => +1
        if axis_idx == 0:
            self.mirror[0] = -1.0 if self.chk_mx.isChecked() else 1.0
        elif axis_idx == 1:
            self.mirror[1] = -1.0 if self.chk_my.isChecked() else 1.0
        elif axis_idx == 2:
            self.mirror[2] = -1.0 if self.chk_mz.isChecked() else 1.0
        self.aplicar_transformacao()

    def reset_pose(self):
        self.offset[:] = 0.0
        self.rotacao[:] = 0.0
        for k in ("tx","ty","tz","rx","ry","rz"):
            if k in self.spins:
                self.spins[k].setValue(0.0)  # se você renomear, remova isto
        # Caso tenha usado labels t/r xyz simples:
        for eixo in "xyz":
            self.spins[f"t{eixo}"].setValue(0.0)
            self.spins[f"r{eixo}"].setValue(0.0)
        self.aplicar_transformacao()

    def reset_mirror(self):
        self.mirror[:] = 1.0
        self.chk_mx.setChecked(False)
        self.chk_my.setChecked(False)
        self.chk_mz.setChecked(False)
        self.aplicar_transformacao()

    # --- Aplicar (sempre a partir do original, sem acúmulo) ---
    def aplicar_transformacao(self):
        """
        P2 = R · ( S · (P0 - c0) ) + c0 + t
        S = diag([mx,my,mz]) com mx,my,mz ∈ {+1, -1}
        """
        R_mat = R.from_euler('xyz', self.rotacao, degrees=True).as_matrix()

        # aplica espelho local e rotação ao redor do pivot
        P_local = (self.P0 - self.c0) * self.mirror      # S · (P0 - c0)
        P2 = (R_mat @ P_local.T).T + self.c0 + self.offset

        # Atualiza pontos in-place (estável) e marca como modificado
        self.malha_vtu.points = P2
        try:
            self.malha_vtu.modified()
        except Exception:
            pass

        # Se a paridade do espelho mudou (produto muda sinal), apenas flip nas normais existentes
        try:
            current_sign = float(np.prod(self.mirror))
            if np.sign(current_sign) != np.sign(self.last_mirror_sign):
                nrm = self.malha_vtu.point_data.get('Normals', None)
                if nrm is not None:
                    self.malha_vtu.point_data['Normals'] = -nrm
                self.last_mirror_sign = current_sign
        except Exception:
            pass

        self.plotter.render()

    # --- Salvar transformação atual (móvel → fixo) ---
    def salvar_transformados(self):
        """
        Salva R, S (espelhos), t e T/T_inv montadas com o mesmo pivot c0 usado na visualização:
        T = T_t · T_p · (R · S) · T_-p
        """
        R_mat = R.from_euler('xyz', self.rotacao, degrees=True).as_matrix()
        S_mat = np.diag(self.mirror.astype(float))   # S (3x3) com ±1
        RS = R_mat @ S_mat                           # composição linear final
        t_vec = self.offset.copy()
        pivot = self.c0.copy()

        # T = T_t · T_p · (R·S) · T_-p
        T_core = np.eye(4)
        T_core[:3, :3] = RS
        T_p  = np.eye(4);  T_p[:3, 3]  = pivot
        T_np = np.eye(4);  T_np[:3, 3] = -pivot
        T_t  = np.eye(4);  T_t[:3, 3]  = t_vec
        T = T_t @ T_p @ T_core @ T_np
        T_inv = npl.inv(T)

        det_RS = float(npl.det(RS))
        meta = {
            "is_rigid": True,
            "has_reflection": (det_RS < 0.0),
            "euler_order": "xyz",
            "pivot_used": pivot.tolist(),
            "mirror_axes": self.mirror.tolist(),     # [+1/-1 por eixo]
            "S_3x3": S_mat.tolist(),
            "R_3x3": R_mat.tolist(),
            "RS_3x3": RS.tolist(),
            "det_RS": det_RS,
            "t_3": t_vec.tolist(),
            "compose_equation": "T = T_t · T_p · (R·S) · T_-p",
            "T_4x4": T.tolist(),
            "T_inv_4x4": T_inv.tolist()
        }

        with open("transformacao_rigida.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        np.save("T_4x4.npy", T)
        np.save("T_inv_4x4.npy", T_inv)
        print("[OK] Transformação salva em 'transformacao_rigida.json', 'T_4x4.npy', 'T_inv_4x4.npy'.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    janela = JanelaControle()
    janela.resize(1200, 900)
    janela.show()
    sys.exit(app.exec_())
