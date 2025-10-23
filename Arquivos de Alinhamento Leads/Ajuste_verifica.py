import sys
import json
import numpy as np
import numpy.linalg as npl
import pandas as pd
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QGridLayout, QMessageBox
)
from PyQt5.QtCore import Qt


# ---------- Eletrodos de fallback (mm) ----------
LEADS_AUTOMATICO_MM = np.array(
    [
        [25455.1, -24790.8, 128147.5],  # V1
        [63604.6, -25589.5, 121263.0],  # V2
        [93223.7, -26570.4, 104170.7],  # V3
        [119019.3, -25818.6, 84240.7],  # V4
        [189466.4, 21790.6, 88558.1],  # V5
        [221677.3, 89290.6, 95483.6],  # V6
        [193603.4, 76137.2, 274150.0],  # LA
        [176445.3, 18737.4, -107207.8],  # LL
        [-77499.5, 2171.4, -98577.8],  # RL
        [-99483.4, 73894.1, 274536.3],  # RA
    ], dtype=np.float64
)

# Se quiser, substitua esse fallback pelos seus pontos manuais
LEADS_REAL_MM = np.array(
    [
        [ 28550.,  -25192., 121103.],  # V1
        [ 73790.,  -25372., 121963.],  # V2
        [104110.,  -25512., 103413.],  # V3
        [131180.,  -20092.,  84203.],  # V4
        [219310.,   80358.,  86233.],  # V5
        [248180.,  118848.,  89693.],  # V6
        [236530.,   67408., 179573.],  # LA
        [186530.,   67408.,-420427.],  # LL
        [ -88390.,   67408.,-420427.], # RL
        [-138390.,   67408., 179573.], # RA
    ], dtype=np.float64
)


# ---------- Funções utilitárias ----------
def apply_T(points_xyz_m: np.ndarray, T44: np.ndarray) -> np.ndarray:
    """Aplica uma matriz homogênea 4x4 a pontos (N,3) em metros."""
    P_h = np.c_[points_xyz_m, np.ones((points_xyz_m.shape[0], 1))]
    return (T44 @ P_h.T).T[:, :3]


def load_electrodes_mm(path_txt: str | None, fallback_mm: np.ndarray) -> np.ndarray:
    """Lê eletrodos de um .txt (x y z em mm) e retorna (N,3) em METROS. Se falhar, usa fallback_mm."""
    if path_txt:
        try:
            arr = np.loadtxt(path_txt, comments="#")
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] < 3:
                raise ValueError("Arquivo de eletrodos deve ter 3 colunas (x y z).")
            return arr[:, :3].astype(float) / 1000.0
        except Exception as e:
            print(f"[Aviso] Falha ao ler '{path_txt}'. Usando fallback. Detalhe: {e}")
    return fallback_mm.astype(float) / 1000.0


# ---------- Classe principal ----------
class JanelaControle(QWidget):
    def __init__(
        self,
        path_alg="Coração .Alg/Patient_1.alg",
        path_coracao_movel="Segmentações/Segmentação Original/coração Abhi.vtp",
        path_torso="Segmentações/Segmentação Original/torso.vtp",
        # Novos caminhos opcionais:
        path_eletrodos_manual_txt=None,   # .txt (mm) MANUAL
        path_eletrodos_auto_txt=None,     # .txt (mm) AUTO
        path_json="Arquivos de Alinhamento Leads/transformacao_rigida.json",
    ):
        super().__init__()
        self.setWindowTitle("Aplicar Transformação: Móvel → Fixo (.alg)")

        self.path_alg = path_alg
        self.path_coracao_movel = path_coracao_movel
        self.path_torso = path_torso
        self.path_eletrodos_manual_txt = path_eletrodos_manual_txt
        self.path_eletrodos_auto_txt = path_eletrodos_auto_txt
        self.path_json = path_json

        # --- Layout e PyVista ---
        layout = QVBoxLayout()
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        self.lbl_status = QLabel("Carregando arquivos…")
        self.lbl_status.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.lbl_status)

        grid = QGridLayout()
        btn_reload = QPushButton("Recarregar & Reaplicar T")
        btn_reload.clicked.connect(self.reload_and_apply)
        grid.addWidget(btn_reload, 0, 0)

        btn_salvar = QPushButton("Salvar Transformados")
        btn_salvar.clicked.connect(self.salvar_transformados)
        grid.addWidget(btn_salvar, 0, 1)

        layout.addLayout(grid)
        self.setLayout(layout)

        # Dados internos
        self.pontos_alg = None

        self.malha_coracao_original = None
        self.malha_coracao = None

        self.malha_torso_original = None
        self.malha_torso = None

        # Eletrodos
        self.leads_manual_original_m = None
        self.leads_auto_original_m = None
        self.leads_manual_m = None
        self.leads_auto_m = None

        self.T = None

        # Executa pipeline inicial
        self.reload_and_apply()


    # ---------- Pipeline principal ----------
    def reload_and_apply(self):
        try:
            self.plotter.clear()

            # --- 1) .ALG fixo ---
            df = pd.read_csv(self.path_alg, header=None)
            points_alg_m = df[[0, 1, 2]].values.astype(float) / 1000.0
            self.pontos_alg = pv.PolyData(points_alg_m)
            if df.shape[1] >= 17:
                valores = df.iloc[:, 8:17].values.astype(float)
                for i in range(valores.shape[1]):
                    self.pontos_alg[f"campo_{8+i}"] = valores[:, i]

            # --- 2) Coração móvel ---
            self.malha_coracao_original = pv.read(self.path_coracao_movel)
            self.malha_coracao = self.malha_coracao_original.copy()

            # --- 3) Torso ---
            self.malha_torso_original = pv.read(self.path_torso)
            self.malha_torso = self.malha_torso_original.copy()

            # --- 4) Eletrodos (MANUAL + AUTO) ---
            self.leads_manual_original_m = load_electrodes_mm(
                self.path_eletrodos_manual_txt, LEADS_REAL_MM
            )
            self.leads_auto_original_m = load_electrodes_mm(
                self.path_eletrodos_auto_txt, LEADS_AUTOMATICO_MM
            )
            # Copias que serão transformadas
            self.leads_manual_m = self.leads_manual_original_m.copy()
            self.leads_auto_m = self.leads_auto_original_m.copy()

            # --- 5) Carrega T do JSON ---
            with open(self.path_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            T = np.array(data["T_4x4"], dtype=float)
            if T.shape != (4, 4):
                raise ValueError("T_4x4 no JSON não é 4x4.")
            self.T = T

            # --- 6) Aplica T no coração e torso ---
            self.malha_coracao.points = apply_T(self.malha_coracao_original.points, self.T)
            self.malha_torso.points = apply_T(self.malha_torso_original.points, self.T)

            # --- 7) Calcula centro transformado do coração ---
            R_mat = self.T[:3, :3]
            t_vec = self.T[:3, 3]
            centro_original = np.array(self.malha_coracao_original.center)
            centro_transformado = (R_mat @ centro_original) + t_vec

            # --- 8) Aplica rotação/translação em torno desse centro aos eletrodos (MANUAL e AUTO) ---
            self.leads_manual_m = (R_mat @ (self.leads_manual_original_m - centro_transformado).T).T + centro_transformado
            self.leads_auto_m   = (R_mat @ (self.leads_auto_original_m   - centro_transformado).T).T + centro_transformado

            # --- 9) Plotagem ---
            self.plotter.add_mesh(
                self.pontos_alg,
                scalars=("campo_8" if "campo_8" in self.pontos_alg.array_names else None),
                color=("red" if "campo_8" not in self.pontos_alg.array_names else None),
                point_size=6, render_points_as_spheres=True
            )

            self.plotter.add_mesh(self.malha_coracao, color="white", opacity=0.7, show_edges=True)
            self.plotter.add_mesh(self.malha_torso, color="lightgray", opacity=0.3, show_edges=True)

            # Eletrodos: MANUAL (laranja) e AUTO (azul)
            self.plotter.add_mesh(
                pv.PolyData(self.leads_manual_m),
                color="orange", point_size=12, render_points_as_spheres=True
            )
            self.plotter.add_mesh(
                pv.PolyData(self.leads_auto_m),
                color="blue", point_size=12, render_points_as_spheres=True
            )

            # Legenda
            self.plotter.add_legend(
                labels=[
                    ("ALG (.alg)", "red"),
                    ("Coração (T aplicado)", "white"),
                    ("Torso (T aplicado)", "lightgray"),
                    ("Leads REAL", "orange"),
                    ("Leads AUTO", "blue"),
                ],
                bcolor="black", border=True, loc="upper right"
            )

            self.plotter.show_axes()
            self.plotter.show_grid()
            self.plotter.reset_camera()

            self.lbl_status.setText("OK: transformação aplicada (REAL + AUTO) com rotação em torno do coração transformado.")

        except Exception as e:
            self.lbl_status.setText(f"Erro: {e}")
            QMessageBox.critical(self, "Erro", str(e))


# ---------- Salvar arquivos transformados ----------
    def salvar_transformados(self):
        import os

        try:
            if self.malha_coracao is None or self.malha_torso is None:
                raise RuntimeError("Nada para salvar. Recarregue e aplique a transformação primeiro.")

            if self.leads_manual_m is None or self.leads_auto_m is None:
                raise RuntimeError("Eletrodos não disponíveis. Recarregue e aplique a transformação.")

            # --- pasta de saída ---
            pasta_out = "Arquivos de Alinhamento Leads"
            os.makedirs(pasta_out, exist_ok=True)

            # --- salva eletrodos em mm ---
            np.savetxt(
                os.path.join(pasta_out, "leads_manual_transformados.txt"),
                self.leads_manual_m * 1000.0,
                fmt="%.6f",
                header="Eletrodos REAIS transformados (mm): x y z",
                comments=""
            )
            np.savetxt(
                os.path.join(pasta_out, "leads_auto_transformados.txt"),
                self.leads_auto_m * 1000.0,
                fmt="%.6f",
                header="Eletrodos AUTOMÁTICOS transformados (mm): x y z",
                comments=""
            )

            self.lbl_status.setText(
                f"Arquivos salvos em '{pasta_out}': coracao_transformado.vtp, "
                "torso_transformado.vtp, leads_manual_transformados.txt, leads_auto_transformados.txt"
            )
            QMessageBox.information(self, "Salvar", f"Arquivos salvos em:\n{pasta_out}")

        except Exception as e:
            self.lbl_status.setText(f"Erro ao salvar: {e}")
            QMessageBox.critical(self, "Erro ao salvar", str(e))


# ---------- Execução ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    janela = JanelaControle(
        path_alg="Coração .Alg/Patient_1.alg",
        path_coracao_movel="Segmentações/Segmentação Original/coração Abhi.vtp",
        path_torso="Segmentações/Segmentação Original/torso.vtp",
        # Preencha estes dois se tiver os arquivos .txt (em mm):
        path_eletrodos_manual_txt=None,    # ex: "leads_man.txt"
        path_eletrodos_auto_txt=None,      # ex: "leads_auto.txt"
        path_json="Arquivos de Alinhamento Leads/transformacao_rigida.json",
    )
    janela.resize(1200, 900)
    janela.show()
    sys.exit(app.exec_())
