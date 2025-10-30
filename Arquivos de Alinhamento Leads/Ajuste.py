import sys
import os
import json
import numpy as np
import numpy.linalg as npl
import pandas as pd
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QDoubleSpinBox,
    QPushButton, QGridLayout, QCheckBox, QHBoxLayout
)
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree


# =========================
# ======= PARÂMETROS ======
# =========================
# Amostragem das nuvens
N_SAMPLES_VTK_DEFAULT = 150_000   # pontos da malha .vtp (convertida para point cloud)
N_SAMPLES_ALG_DEFAULT = 200_000   # pontos do .alg (pode reduzir se pesado)

# ICP - valores padrão (ajustáveis na UI)
ICP_ITERS_DEFAULT      = 40
ICP_TRIM_RATIO_DEFAULT = 0.90     # manter piores 10% fora (0<r<=1)
ICP_CLIP_PCT_DEFAULT   = 95.0     # descartar correspondências acima do percentil de distância

# ===== Degradê (visibilidade) =====
# Aceleração do degradê: >1 = muda mais rápido com a profundidade
FADE_GAMMA = 2.2         # experimente 1.6–3.0
ALPHA_MIN  = 80          # 0..255 (mais baixo = mais transparente no "fundo")
ALPHA_MAX  = 255         # 0..255 (mais alto = mais opaco no "perto")

# Paleta de cores para o degradê estático
# Opções: "green_yellow", "blue_red", "grayscale"
COLOR_PALETTE = "green_yellow"

# =========================
# ======= FUNÇÕES =========
# =========================

def kabsch_R_t(A: np.ndarray, B: np.ndarray):
    """
    Resolve R, t minimizando || R A + t - B ||_F (ponto-a-ponto).
    A, B: Nx3
    Retorna: R(3x3), t(3,)
    """
    if len(A) == 0:
        return np.eye(3), np.zeros(3)
    muA = A.mean(axis=0)
    muB = B.mean(axis=0)
    A0 = A - muA
    B0 = B - muB
    H = A0.T @ B0
    U, S, Vt = npl.svd(H)
    Rm = Vt.T @ U.T
    # Corrige reflexão: força det(R)=+1 (espelho é tratado em S, não na rotação)
    if npl.det(Rm) < 0:
        Vt[-1, :] *= -1
        Rm = Vt.T @ U.T
    t = muB - Rm @ muA
    return Rm, t


def apply_pose_to_points(P0: np.ndarray, c0: np.ndarray,
                         R_mat: np.ndarray, t_vec: np.ndarray, S_vec: np.ndarray) -> np.ndarray:
    """
    Aplica pose aos pontos de referência P0 usando pivô dinâmico (c0 + t):
    P = R · ( S · (P0 - c0) ) + (c0 + t)
    """
    pivot = c0 + t_vec
    S = np.diag(S_vec.astype(float))
    P_local = (P0 - c0) @ S.T
    return (R_mat @ P_local.T).T + pivot


def euler_from_R_xyz(R_mat: np.ndarray) -> np.ndarray:
    return R.from_matrix(R_mat).as_euler('xyz', degrees=True)


def clamp_angle180(a):
    a = (a + 180.0) % 360.0 - 180.0
    return a


class JanelaControle(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Alinhamento (ICP) - VTP como Nuvem vs .ALG (fixo)")

        layout = QVBoxLayout()
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        # --- Carrega .alg (FIXO) ---
        df = pd.read_csv("Coração .Alg/Patient_10.alg", header=None)
        points_alg_all = df[[0, 1, 2]].values.astype(float) / 1000.0
        if points_alg_all.shape[0] > N_SAMPLES_ALG_DEFAULT:
            idx = np.random.choice(points_alg_all.shape[0], N_SAMPLES_ALG_DEFAULT, replace=False)
            self.points_alg = points_alg_all[idx]
            valores = df.iloc[idx, 8:17].values.astype(float)
        else:
            self.points_alg = points_alg_all
            valores = df.iloc[:, 8:17].values.astype(float)

        self.pontos_alg = pv.PolyData(self.points_alg)
        for i in range(valores.shape[1]):
            self.pontos_alg[f"campo_{8+i}"] = valores[:, i]
        
        # --- Carrega malha .vtp (MÓVEL) e converte para point cloud ---
        vtk_mesh = pv.read("Segmentações/Segmentação Original/Coração Abhi.vtp")
        self.P0_mesh = vtk_mesh.points.copy()                 # referência
        self.c0 = np.array(vtk_mesh.center, dtype=float)      # centro da malha

        # Amostra vértices (ou use pontos na superfície se quiser)
        P_all = self.P0_mesh
        if P_all.shape[0] > N_SAMPLES_VTK_DEFAULT:
            idx = np.random.choice(P_all.shape[0], N_SAMPLES_VTK_DEFAULT, replace=False)
            self.P0 = P_all[idx].copy()
        else:
            self.P0 = P_all.copy()

        # --- Estado (pose absoluta): R (3x3), t (3,), S (±1) ---
        self.offset  = np.zeros(3, dtype=float)               # t
        self.R_mat   = np.eye(3, dtype=float)                 # R
        self.rotacao = np.zeros(3, dtype=float)               # Euler xyz (°) – UI
        self.mirror  = np.array([1.0, 1.0, 1.0], dtype=float) # S (±1)

        # --- PolyData do VTK transformado (para render) ---
        self.P_poly = pv.PolyData(self.P0.copy())  # começa sem transformação

        # Plota ALGs (fixo)
        self.plotter.add_mesh(
            self.pontos_alg,
            scalars="campo_8",
            render_points_as_spheres=True,
            point_size=5.0,
            color='red',
            lighting=False,
            name="ALG"
        )

        # Placeholder do escalar de degradê (será preenchido uma vez após reset_camera)
        # Cria um array RGBA fixo (substituído depois pelo degradê estático)
        rgba_init = np.full((self.P_poly.n_points, 4), [154, 160, 166, 200], dtype=np.uint8)
        self.P_poly["rgba"] = rgba_init

        self.actor_pts = self.plotter.add_mesh(
            self.P_poly,
            scalars="rgba",
            rgba=True,                       # usa as cores RGBA diretamente (sem cmap)
            render_points_as_spheres=True,
            point_size=4.0,
            lighting=False,
            name="VTK_PTS"
        )


        # --- UI: Translação e Rotação ---
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

        # --- Parâmetros de ICP + Botão ---
        icp_row = QHBoxLayout()
        icp_row.addWidget(QLabel("ICP iters"))
        self.spin_icp_iters = QDoubleSpinBox(); self.spin_icp_iters.setDecimals(0)
        self.spin_icp_iters.setRange(1, 500); self.spin_icp_iters.setValue(ICP_ITERS_DEFAULT)
        icp_row.addWidget(self.spin_icp_iters)

        icp_row.addWidget(QLabel("VTK amostras"))
        self.spin_icp_samp = QDoubleSpinBox(); self.spin_icp_samp.setDecimals(0)
        self.spin_icp_samp.setRange(1000, 1_000_000); self.spin_icp_samp.setValue(N_SAMPLES_VTK_DEFAULT)
        icp_row.addWidget(self.spin_icp_samp)

        icp_row.addWidget(QLabel("trim"))
        self.spin_icp_trim = QDoubleSpinBox(); self.spin_icp_trim.setDecimals(2)
        self.spin_icp_trim.setRange(0.10, 1.00); self.spin_icp_trim.setValue(ICP_TRIM_RATIO_DEFAULT)
        icp_row.addWidget(self.spin_icp_trim)

        icp_row.addWidget(QLabel("clip %"))
        self.spin_icp_clip = QDoubleSpinBox(); self.spin_icp_clip.setDecimals(1)
        self.spin_icp_clip.setRange(50.0, 100.0); self.spin_icp_clip.setValue(ICP_CLIP_PCT_DEFAULT)
        icp_row.addWidget(self.spin_icp_clip)

        self.btn_icp = QPushButton("Rodar ICP")
        self.btn_icp.clicked.connect(self.rodar_icp)
        icp_row.addWidget(self.btn_icp)
        layout.addLayout(icp_row)

        # --- Botões utilitários ---
        btn_reset_pose = QPushButton("Resetar Pose (R, t)")
        btn_reset_pose.clicked.connect(self.reset_pose)
        layout.addWidget(btn_reset_pose)

        btn_reset_mirror = QPushButton("Resetar Espelhos")
        btn_reset_mirror.clicked.connect(self.reset_mirror)
        layout.addWidget(btn_reset_mirror)

        btn_salvar = QPushButton("Salvar Transformação")
        btn_salvar.clicked.connect(self.salvar_transformados)
        layout.addWidget(btn_salvar)

        self.setLayout(layout)

        # --- Render inicial ---
        self.plotter.show_axes()
        self.plotter.show_grid()
        self.plotter.reset_camera()

        # Aplica a pose inicial (identidade)
        self.aplicar_transformacao()

        # Calcula o degradê UMA VEZ com base na câmera atual e congela (não atualiza mais)  ### NOVO
        self._calcular_fade_estatico()
        self.plotter.render()

    # =========================
    # ===== Callbacks UI ======
    # =========================
    def atualizar_translacao(self, eixo, novo_valor):
        self.offset[eixo] = float(novo_valor)
        self.aplicar_transformacao()

    def atualizar_rotacao(self, eixo, novo_valor):
        self.rotacao[eixo] = float(novo_valor)
        self.R_mat = R.from_euler('xyz', self.rotacao, degrees=True).as_matrix()
        self.aplicar_transformacao()

    def toggle_mirror(self, axis_idx: int):
        if axis_idx == 0:
            self.mirror[0] = -1.0 if self.chk_mx.isChecked() else 1.0
        elif axis_idx == 1:
            self.mirror[1] = -1.0 if self.chk_my.isChecked() else 1.0
        elif axis_idx == 2:
            self.mirror[2] = -1.0 if self.chk_mz.isChecked() else 1.0
        self.aplicar_transformacao()

    def reset_pose(self):
        self.offset[:] = 0.0
        self.R_mat = np.eye(3)
        self.rotacao[:] = 0.0
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

    # =========================
    # == Aplicar Transform ====
    # =========================
    def aplicar_transformacao(self):
        """
        Rotação estável em torno do próprio eixo:
        P = R · ( S · (P0 - c0) ) + (c0 + t)
        onde o pivô dinâmico é (c0 + t).
        """
        P = apply_pose_to_points(self.P0, self.c0, self.R_mat, self.offset, self.mirror)
        # Atualiza PolyData in-place
        self.P_poly.points = P
        try:
            self.P_poly.modified()
        except Exception:
            pass
        self.plotter.render()
        # OBS: não recalculamos o 'fade' aqui — ele é estático (congelado)  ### NOVO

    # =========================
    # ======== ICP ============
    # =========================
    def rodar_icp(self):
        # Ajustes da UI
        n_iters = int(self.spin_icp_iters.value())
        n_samp  = int(self.spin_icp_samp.value())
        trim_r  = float(self.spin_icp_trim.value())
        clip_pct = float(self.spin_icp_clip.value())

        # Constrói KDTree do alvo (.alg)
        alg_pts = self.points_alg
        kdt = cKDTree(alg_pts)

        # Fonte atual (já com pose/mirror aplicados)
        P_src = apply_pose_to_points(self.P0, self.c0, self.R_mat, self.offset, self.mirror)

        # Amostra a fonte para ICP (performance)
        if P_src.shape[0] > n_samp:
            idx_s = np.random.choice(P_src.shape[0], n_samp, replace=False)
            P_src = P_src[idx_s]
            P0_sub = self.P0[idx_s]
        else:
            P0_sub = self.P0  # usado depois para recompor a pose

        # ICP iterativo: encontra ΔR, Δt em espaço global e acumula
        R_acc = np.eye(3)
        t_acc = np.zeros(3)

        for it in range(n_iters):
            # Aplica incremento atual à nuvem fonte corrente
            P_it = (R_acc @ P_src.T).T + t_acc

            # Correspondências
            dists, nn_idx = kdt.query(P_it, k=1, workers=-1)
            # Clipping por percentil
            if 0.0 < clip_pct < 100.0:
                thresh = np.percentile(dists, clip_pct)
                mask = dists <= thresh
                P_fit = P_it[mask]
                Q_fit = alg_pts[nn_idx[mask]]
            else:
                P_fit = P_it
                Q_fit = alg_pts[nn_idx]

            # Trim ratio
            if 0.0 < trim_r < 1.0 and P_fit.shape[0] > 0:
                m = int(np.floor(trim_r * P_fit.shape[0]))
                if m >= 6:
                    order = np.argsort(np.sum((P_fit - Q_fit)**2, axis=1))
                    sel = order[:m]
                    P_fit = P_fit[sel]
                    Q_fit = Q_fit[sel]

            if P_fit.shape[0] < 6:
                break  # muito poucos pontos para SVD

            # Resolve ΔR, Δt (alinha P_it -> Q)
            dR, dt = kabsch_R_t(P_fit, Q_fit)
            # Acumula: P_new = dR * P_it + dt = (dR*R_acc) * P_src + (dR*t_acc + dt)
            R_acc = dR @ R_acc
            t_acc = dR @ t_acc + dt

        # Resultado incremental aplicado na pose global atual
        P_final = (R_acc @ apply_pose_to_points(self.P0, self.c0, self.R_mat, self.offset, self.mirror).T).T + t_acc

        # Converte para (R_new, t_new) no nosso modelo
        S_mat = np.diag(self.mirror.astype(float))
        A = (self.P0 - self.c0) @ S_mat.T
        B = P_final - self.c0
        if A.shape[0] > n_samp:
            idx = np.random.choice(A.shape[0], n_samp, replace=False)
            A_fit = A[idx]
            B_fit = B[idx]
        else:
            A_fit = A
            B_fit = B

        R_new, t_new_rel = kabsch_R_t(A_fit, B_fit)
        t_new = t_new_rel.copy()

        # Atualiza estado + UI
        self.R_mat = R_new
        eul = euler_from_R_xyz(R_new)
        self.rotacao[:] = [clamp_angle180(a) for a in eul]
        self.offset[:] = t_new

        for i, eixo in enumerate("xyz"):
            self.spins[f"r{eixo}"].blockSignals(True)
            self.spins[f"t{eixo}"].blockSignals(True)
            self.spins[f"r{eixo}"].setValue(float(self.rotacao[i]))
            self.spins[f"t{eixo}"].setValue(float(self.offset[i]))
            self.spins[f"r{eixo}"].blockSignals(False)
            self.spins[f"t{eixo}"].blockSignals(False)

        self.aplicar_transformacao()

    # =========================
    # ======= SALVAR ==========
    # =========================
    def salvar_transformados(self):
        """
        Salva R, S (espelhos), t e T/T_inv montadas com pivô dinâmico c0+t:
        Modelo da aplicação:
        P = R · (S · (P0 - c0)) + (c0 + t)
        A matriz homogênea equivalente (sobre P0-ponto) depende do pivô (c0+t).
        """
        out_dir = os.path.join("Arquivos de Alinhamento Leads")
        os.makedirs(out_dir, exist_ok=True)

        pivot = self.c0 + self.offset
        S_mat = np.diag(self.mirror.astype(float))
        RS = self.R_mat @ S_mat

        # T = T_(pivot) · [RS|0;0|1] · T_-c0
        T_piv = np.eye(4);  T_piv[:3, 3]  = pivot
        T_m_c0 = np.eye(4); T_m_c0[:3, 3] = -self.c0
        T_lin = np.eye(4);  T_lin[:3, :3] = RS
        T = T_piv @ T_lin @ T_m_c0

        T_inv = npl.inv(T)
        det_RS = float(npl.det(RS))

        meta = {
            "is_rigid": True,
            "has_reflection": (det_RS < 0.0),
            "euler_order": "xyz",
            "pivot_used": pivot.tolist(),
            "mirror_axes": self.mirror.tolist(),
            "S_3x3": S_mat.tolist(),
            "R_3x3": self.R_mat.tolist(),
            "RS_3x3": RS.tolist(),
            "det_RS": det_RS,
            "t_3": self.offset.tolist(),
            "compose_equation": "P = R · (S · (P0 - c0)) + (c0 + t)",
            "T_4x4": T.tolist(),
            "T_inv_4x4": T_inv.tolist()
        }

        json_path = os.path.join(out_dir, "transformacao_rigida.json")
        T_path    = os.path.join(out_dir, "T_4x4.npy")
        Tinv_path = os.path.join(out_dir, "T_inv_4x4.npy")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        np.save(T_path, T)
        np.save(Tinv_path, T_inv)

        print(f"[OK] Transformação salva em:\n  {json_path}\n  {T_path}\n  {Tinv_path}")

    # =========================
    # == Degradê estático =====
    # =========================
    def _calcular_fade_estatico(self):
        """
        Calcula um degradê fixo em RGBA por profundidade ao longo do eixo de visão atual.
        Perto = mais opaco e mais "quente"; longe = mais transparente e mais "frio".
        A velocidade do degradê é controlada por FADE_GAMMA (>1 acelera).
        """
        cam = self.plotter.camera
        cam_pos = np.array(cam.position, dtype=float)
        cam_foc = np.array(cam.focal_point, dtype=float)
        view_dir = cam_foc - cam_pos
        nrm = np.linalg.norm(view_dir)

        if nrm < 1e-9 or self.P_poly.n_points == 0:
            # fallback: cinza médio translúcido
            rgba = np.full((self.P_poly.n_points, 4), [180, 180, 180, 180], dtype=np.uint8)
            self.P_poly["rgba"] = rgba
            try: self.P_poly.modified()
            except Exception: pass
            return

        view_dir /= nrm
        P = self.P_poly.points
        depths = (P - cam_pos) @ view_dir

        dmin, dmax = float(depths.min()), float(depths.max())
        if abs(dmax - dmin) < 1e-9:
            fade = np.ones_like(depths)
        else:
            # perto = 1, longe = 0 (linear)
            fade = 1.0 - (depths - dmin) / (dmax - dmin)

        # Ajusta “velocidade” (contraste) do degradê
        fade = np.clip(fade, 0.0, 1.0) ** float(FADE_GAMMA)

        # Alpha com faixa configurável
        alpha = (ALPHA_MIN + fade * (ALPHA_MAX - ALPHA_MIN)).astype(np.uint8)

        # ===== Paletas de cor (RGB) =====
        # green_yellow: longe = verde escuro, perto = amarelo vivo
        if COLOR_PALETTE == "green_yellow":
            # cores em float 0..1 para interpolar
            c_far = np.array([ 30, 119,  58], dtype=np.float32) / 255.0  # verde
            c_near= np.array([253, 231,  37], dtype=np.float32) / 255.0  # amarelo
        elif COLOR_PALETTE == "blue_red":
            c_far = np.array([ 33,  97, 174], dtype=np.float32) / 255.0  # azul
            c_near= np.array([220,  33,  38], dtype=np.float32) / 255.0  # vermelho
        else:  # "grayscale"
            c_far = np.array([ 80,  80,  80], dtype=np.float32) / 255.0
            c_near= np.array([240, 240, 240], dtype=np.float32) / 255.0

        # Interpola cores por fade (broadcasting)
        fade_f = fade.astype(np.float32)[:, None]  # (N,1)
        rgb = (c_far * (1.0 - fade_f) + c_near * fade_f)
        rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

        rgba = np.column_stack([rgb_u8, alpha]).astype(np.uint8)
        self.P_poly["rgba"] = rgba
        try:
            self.P_poly.modified()
        except Exception:
            pass




if __name__ == "__main__":
    app = QApplication(sys.argv)
    janela = JanelaControle()
    janela.resize(1200, 900)
    janela.show()
    sys.exit(app.exec_())
