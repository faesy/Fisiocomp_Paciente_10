# salvar como: converter_rois_para_vtp.py
import json
import numpy as np
import pyvista as pv
from pathlib import Path

# ===== Configuração =====
PASTA_ENTRADA = Path("linhas json")   # onde estão .mrk/.json
PASTA_SAIDA   = Path("linhas vtp")    # será criada
MANTER_EM_LPS = True                  # True = mantém LPS; False = converte p/ RAS

def lps_to_ras(p):
    p = np.asarray(p, float).copy()
    p[:2] *= -1.0
    return p

def load_markups(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    # Em versões recentes a chave é "markups"
    nodes = data.get("markups", data.get("Markups", []))
    return nodes

def roi_nodes(nodes):
    for node in nodes:
        if (node.get("type") or node.get("markupType") or "").upper() == "ROI":
            yield node

def parse_roi(node):
    name = node.get("name") or "ROI"
    coord_sys = (node.get("coordinateSystem") or node.get("CoordinateSystem") or "RAS").upper()
    center = np.array(node["center"], float)
    size   = np.array(node["size"],   float)  # comprimentos (mm)
    # orientation: 9 números (linha 1 | linha 2 | linha 3)
    ori = np.array(node.get("orientation", [1,0,0, 0,1,0, 0,0,1]), float).reshape(3,3)
    return name, coord_sys, center, size, ori

def oriented_box_polydata(center, size, R):
    """Gera PolyData de um paralelepípedo orientado.
       center: (3,), size: (3,) comprimentos, R: matriz 3x3 (linhas = eixos mundo de x,y,z locais)."""
    # Meias-dimensões
    hx, hy, hz = size * 0.5

    # Eixos locais no mundo: tomamos as COLUNAS como direções (x,y,z) locais
    # O JSON do Slicer exporta linha-a-linha; para usar colunas como eixos, transponha.
    axes = R.T.copy()
    # normalizar eixos (por segurança)
    axes = np.array([a / (np.linalg.norm(a) + 1e-15) for a in axes])

    ex, ey, ez = axes  # vetores 3D

    # 8 cantos: combinações de sinais
    signs = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
    ], float)

    pts = []
    for sx, sy, sz in signs:
        p = (center
             + sx * hx * ex
             + sy * hy * ey
             + sz * hz * ez)
        pts.append(p)
    pts = np.array(pts, float)

    # Faces (6 quads) usando os índices dos pontos acima
    # Base z- (0,1,2,3), topo z+ (4,5,6,7)
    # Laterais conectando as arestas correspondentes
    quads = [
        [0,1,2,3],  # z-
        [4,5,6,7],  # z+
        [0,1,5,4],  # y-
        [1,2,6,5],  # x+
        [2,3,7,6],  # y+
        [3,0,4,7],  # x-
    ]
    faces = []
    for q in quads:
        faces.extend([4, *q])  # cada face: [n=4, i0,i1,i2,i3]
    faces = np.array(faces, np.int64)

    mesh = pv.PolyData(pts, faces)
    mesh.compute_normals(inplace=True)
    return mesh

def main():
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    arquivos = sorted(list(PASTA_ENTRADA.glob("*.mrk")) + list(PASTA_ENTRADA.glob("*.json")))
    if not arquivos:
        print(f"Nenhum .mrk/.json encontrado em: {PASTA_ENTRADA.resolve()}")
        return

    cont = 0
    for arq in arquivos:
        try:
            nodes = load_markups(arq)
        except Exception as e:
            print(f"[ERRO] {arq.name}: não consegui ler JSON ({e})")
            continue

        for idx, node in enumerate(roi_nodes(nodes)):
            try:
                name, coord_sys, center, size, R = parse_roi(node)

                # manter em LPS (sem conversão) ou converter para RAS:
                if not MANTER_EM_LPS:
                    # salvaremos em RAS: converter center LPS->RAS; orientação: inverte sinais de X e Y
                    center = lps_to_ras(center)
                    # Para orientação, mudamos de LPS p/ RAS: aplicar inversão nos eixos X e Y do espaço alvo.
                    # Equivale a: R_ras = D * R_lps * D, com D = diag(-1,-1,1), mas como usamos R apenas para eixos,
                    # transpor para eixos e aplicar sinais nas componentes X e Y:
                    D = np.diag([-1, -1, 1])
                    R = D @ R @ D

                mesh = oriented_box_polydata(center, size, R)
                base = f"{arq.stem}_{name.replace(' ', '_')}_{idx:02d}"
                out_path = PASTA_SAIDA / f"{base}.vtp"
                mesh.save(out_path)
                print(f"✓ {arq.name} :: {name} → {out_path.name}")
                cont += 1
            except Exception as e:
                print(f"[ERRO] {arq.name} (ROI #{idx}): {e}")

    print(f"\nConcluído: {cont} ROI(s) exportados para {PASTA_SAIDA.resolve()}")

if __name__ == "__main__":
    main()
