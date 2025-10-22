import pyvista as pv
import SimpleITK as sitk
import numpy as np
import os

# ========= CONFIGURAÇÕES =========
input_folder = "linhas vtp"        # pasta com os .vtk
output_folder = "linhas_nii"   # pasta para salvar os .nii
voxel_size = 0.25  # mm (ajuste a resolução)

os.makedirs(output_folder, exist_ok=True)

# ========= LOOP SOBRE ARQUIVOS =========
for filename in os.listdir(input_folder):
    if filename.endswith(".vtp"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".vtp", ".nii.gz"))
        
        print(f"Convertendo {filename} ...")

        # Ler a malha
        mesh = pv.read(input_path)
        bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)

        # Definir dimensões do volume
        dims = [
            int((bounds[1] - bounds[0]) / voxel_size),
            int((bounds[3] - bounds[2]) / voxel_size),
            int((bounds[5] - bounds[4]) / voxel_size),
        ]

        if min(dims) <= 0:
            print(f"⚠️  Dimensão inválida para {filename}, pulando...")
            continue

        # Criar grade
        grid = pv.create_grid(mesh, dimensions=dims)

        # Voxelizar
        voxelized = grid.select_enclosed_points(mesh, tolerance=0.0)

        # Criar máscara binária
        mask = voxelized.point_data["SelectedPoints"].reshape(dims, order="F")
        mask = np.transpose(mask, (2, 1, 0))  # ajustar eixos para SimpleITK

        # Converter para SimpleITK
        image = sitk.GetImageFromArray(mask.astype(np.uint8))
        image.SetSpacing([voxel_size, voxel_size, voxel_size])
        image.SetOrigin([bounds[0], bounds[2], bounds[4]])

        # Salvar
        sitk.WriteImage(image, output_path)
        print(f"✅ Salvo: {output_path}")

print("🚀 Conversão concluída!")
