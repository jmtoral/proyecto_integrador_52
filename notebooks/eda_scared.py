# -*- coding: utf-8 -*-
# %% [markdown]
# # EDA del dataset SCARED
#
# **Análisis exploratorio para reconstrucción 3D endoscópica**
#
# Este notebook hace análisis exploratorio sobre los 5 keyframes que usaremos en los tres métodos
# (NExF, EndoDepthAndMotion+MonoIIF, EndoGaussian). Los outputs van a `./eda_outputs/`.
#
# Si no tienes SCARED descargado todavía, el notebook genera datos sintéticos parecidos
# para que puedas probar el pipeline.

# %% Setup
import os, json, warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from pathlib import Path
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

# %% Paths y constantes
SCARED_ROOT = Path("/data/SCARED_common")   # ← AJUSTA esto a tu ruta real
OUTPUT_DIR  = Path("./eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
KEYFRAMES   = ["d1k1", "d2k1", "d3k1", "d6k1", "d7k1"]

# %% Fallback sintético — útil si aún no tienes SCARED descargado
USE_SYNTHETIC = not SCARED_ROOT.exists()

if USE_SYNTHETIC:
    print(f"⚠️  {SCARED_ROOT} no existe. Generando datos sintéticos para demo del pipeline.")
    SCARED_ROOT = Path("./synthetic_scared")
    SCARED_ROOT.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    H, W = 256, 320

    for kf in KEYFRAMES:
        d = SCARED_ROOT / kf
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "depth").mkdir(exist_ok=True)
        (d / "masks").mkdir(exist_ok=True)

        # número de frames variable por keyframe (como en SCARED real)
        N = int(rng.integers(800, 1200))
        # parámetros que distinguen los keyframes
        depth_center = rng.uniform(60, 120)            # profundidad promedio
        sat_rate     = rng.uniform(0.05, 0.35)         # qué tan especular es la escena
        bright_drift = rng.uniform(-30, 30, size=N)    # cambio de iluminación temporal

        for i in range(N):
            # imagen RGB: tejido base + ruido + vignette + specular ocasional
            base = 100 + bright_drift[i] + rng.normal(0, 25, (H, W, 3))
            # vignette (oscuro en bordes)
            x, y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
            vignette = np.exp(-(x**2 + y**2) * 0.8)
            base *= vignette[..., None]
            img = np.clip(base, 0, 255).astype(np.uint8)
            # specular highlights aleatorios
            if rng.random() < sat_rate:
                for _ in range(rng.integers(1, 4)):
                    cy, cx = rng.integers(20, H-20), rng.integers(20, W-20)
                    r = rng.integers(3, 12)
                    cv2.circle(img, (cx, cy), r, (255, 255, 255), -1)
            cv2.imwrite(str(d / "images" / f"{i:06d}.png"), img)

            # depth: superficie suave en mm + ruido + invalidos en bordes y specular
            depth = depth_center + 30 * np.exp(-(x**2 + y**2) / 0.5) + rng.normal(0, 1.5, (H, W))
            depth[vignette < 0.3] = np.nan                # bordes vignette → invalid
            # introduce invalidos donde hay specular (correlación saturación↔invalidez)
            sat_mask = np.any(img > 240, axis=-1)
            depth[sat_mask] = np.nan
            np.save(d / "depth" / f"{i:06d}.npy", depth.astype(np.float32))

            # mask = todo válido (placeholder; en SCARED real generar con SAM2)
            cv2.imwrite(str(d / "masks" / f"{i:06d}.png"),
                        255 * np.ones((H, W), np.uint8))

        # calibración fake (similar a SCARED rectificado)
        K = np.array([[600, 0, W/2], [0, 600, H/2], [0, 0, 1]], dtype=np.float32)
        np.savez(d / "calib.npz",
                 K1=K, K2=K, D1=np.zeros(5), D2=np.zeros(5),
                 R=np.eye(3), T=np.array([[5.4], [0], [0]]))
    print(f"✅ Datos sintéticos generados en {SCARED_ROOT}")

# %% [markdown]
# ## 1. Inventario del dataset
#
# Lo primero: **¿qué tienes?**

# %% Inventario
rows = []
for kf in KEYFRAMES:
    d = SCARED_ROOT / kf
    n_frames = len(list((d / "images").glob("*.png")))
    n_depth  = len(list((d / "depth").glob("*.npy")))
    n_masks  = len(list((d / "masks").glob("*.png")))
    rows.append({"keyframe": kf, "frames": n_frames,
                 "depth_files": n_depth, "mask_files": n_masks})
inventory = pd.DataFrame(rows)
print(inventory.to_string(index=False))
inventory.to_csv(OUTPUT_DIR / "01_inventory.csv", index=False)
total_frames = inventory["frames"].sum()
print(f"\n📊 Total: {total_frames:,} frames en {len(KEYFRAMES)} keyframes")

# %% [markdown]
# ## 2. Estadísticas RGB por keyframe
#
# - **Luminancia media**: cuán brillante es el escenario en promedio.
# - **Std de luminancia**: contraste — escenas muy oscuras o muy uniformes son más difíciles.
# - **% pixeles saturados**: pixeles con cualquier canal RGB > 240; aproxima los specular highlights
#   que rompen la photometric consistency (NExF está diseñado para tolerarlos).

# %% RGB stats — muestreando 1 de cada 5 frames para velocidad
def rgb_stats(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # luminancia BT.601
    lum = 0.299*rgb[..., 0] + 0.587*rgb[..., 1] + 0.114*rgb[..., 2]
    sat = float(np.mean(np.any(rgb > 240, axis=-1)) * 100)
    return dict(mean_lum=float(lum.mean()),
                std_lum=float(lum.std()),
                sat_pct=sat)

rgb_records = []
for kf in KEYFRAMES:
    files = sorted((SCARED_ROOT / kf / "images").glob("*.png"))[::5]
    for i, png in enumerate(files):
        s = rgb_stats(png)
        if s:
            rgb_records.append({"keyframe": kf, "frame": i*5, **s})
df_rgb = pd.DataFrame(rgb_records)
print(df_rgb.groupby("keyframe").agg({"mean_lum":"mean","std_lum":"mean","sat_pct":"mean"}).round(2))

# %% Plot: 3 boxplots lado a lado
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.boxplot(data=df_rgb, x="keyframe", y="mean_lum", ax=axes[0],
            palette="Blues")
axes[0].set_title("Luminancia media (0–255)")
axes[0].set_ylabel("Luminancia"); axes[0].set_xlabel("")
sns.boxplot(data=df_rgb, x="keyframe", y="std_lum", ax=axes[1],
            palette="Greens")
axes[1].set_title("Contraste (std de luminancia)")
axes[1].set_ylabel("Std luminancia"); axes[1].set_xlabel("")
sns.boxplot(data=df_rgb, x="keyframe", y="sat_pct", ax=axes[2],
            palette="Reds")
axes[2].set_title("% pixeles saturados (R/G/B > 240)")
axes[2].set_ylabel("% saturados"); axes[2].set_xlabel("")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_rgb_stats.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close()

# %% [markdown]
# ## 3. Distribución de profundidad (depth GT)
#
# La profundidad GT viene en mm (del structured light de SCARED). Las preguntas:
# - **¿A qué distancia opera el endoscopio?** (cm típicos sobre tejido)
# - **¿Qué fracción de pixeles tienen GT válido?** (los inválidos son NaN o ≤ 0)
# - **¿Hay variación entre keyframes?** (cada anatomía tiene su rango)

# %% Depth stats
def depth_stats(npy_path):
    d = np.load(npy_path)
    valid = ~np.isnan(d) & (d > 0)
    if valid.sum() == 0:
        return None
    dv = d[valid]
    return dict(p5=float(np.percentile(dv, 5)),
                p50=float(np.median(dv)),
                p95=float(np.percentile(dv, 95)),
                d_min=float(dv.min()),
                d_max=float(dv.max()),
                valid_pct=float(valid.mean() * 100))

depth_records = []
for kf in KEYFRAMES:
    files = sorted((SCARED_ROOT / kf / "depth").glob("*.npy"))[::5]
    for i, npy in enumerate(files):
        s = depth_stats(npy)
        if s:
            depth_records.append({"keyframe": kf, "frame": i*5, **s})
df_depth = pd.DataFrame(depth_records)

# %% Plot: violin + box
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.violinplot(data=df_depth, x="keyframe", y="p50", ax=axes[0],
               palette="viridis", inner="quartile")
axes[0].set_title("Profundidad mediana del frame (mm)\n— distancia cámara→tejido")
axes[0].set_ylabel("mm"); axes[0].set_xlabel("")
sns.boxplot(data=df_depth, x="keyframe", y="valid_pct", ax=axes[1],
            palette="magma")
axes[1].set_title("% pixeles con depth GT válido por frame")
axes[1].set_ylabel("% válido"); axes[1].set_xlabel("")
axes[1].set_ylim(0, 105)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_depth_stats.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close()

# %% Histograma global de depth (todos los keyframes mezclados)
print("Calculando histograma global... (puede tardar 30s con SCARED real)")
all_depths = []
for kf in KEYFRAMES:
    for npy in sorted((SCARED_ROOT / kf / "depth").glob("*.npy"))[::20]:
        d = np.load(npy)
        v = d[~np.isnan(d) & (d > 0)]
        # sub-sample para no acumular GBs en RAM
        if len(v) > 5000:
            v = np.random.choice(v, 5000, replace=False)
        all_depths.append(v)
all_depths = np.concatenate(all_depths)

plt.figure(figsize=(11, 4.5))
plt.hist(all_depths, bins=80, color="#2a9d8f", alpha=0.85, edgecolor="white")
plt.axvline(np.median(all_depths), color="crimson", linestyle="--",
            label=f"Mediana = {np.median(all_depths):.1f} mm")
plt.axvline(np.percentile(all_depths, 5), color="orange", linestyle=":",
            label=f"P5 = {np.percentile(all_depths,5):.1f} mm")
plt.axvline(np.percentile(all_depths, 95), color="orange", linestyle=":",
            label=f"P95 = {np.percentile(all_depths,95):.1f} mm")
plt.xlabel("Profundidad (mm)")
plt.ylabel("Frecuencia")
plt.title(f"Distribución global de profundidades GT — N = {len(all_depths):,} pixeles muestreados")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_depth_histogram.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close()

# %% [markdown]
# ## 4. Heatmap espacial: ¿dónde están los pixeles válidos?
#
# Las cámaras endoscópicas tienen **viñeteado** (vignette): los bordes son oscuros y
# frecuentemente sin GT. Un heatmap de "fracción de frames donde este píxel tiene depth válido"
# revela esta forma circular típica.
#
# Esta info es crítica para el cómputo de métricas: **solo se debe evaluar donde hay validez
# espacial razonable** (≥50% típicamente).

# %% Heatmap espacial
fig, axes = plt.subplots(1, len(KEYFRAMES), figsize=(4*len(KEYFRAMES), 4))
for ax, kf in zip(axes, KEYFRAMES):
    files = sorted((SCARED_ROOT / kf / "depth").glob("*.npy"))
    if not files:
        continue
    shape = np.load(files[0]).shape
    valid_count = np.zeros(shape, dtype=np.float32)
    # muestreo cada 10 frames
    sub = files[::10]
    for npy in sub:
        d = np.load(npy)
        valid_count += (~np.isnan(d) & (d > 0)).astype(np.float32)
    valid_frac = valid_count / max(len(sub), 1)
    im = ax.imshow(valid_frac, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(kf); ax.axis("off")
plt.colorbar(im, ax=axes, fraction=0.04, pad=0.02,
             label="Fracción de frames con depth válido")
plt.suptitle("Validez espacial del depth GT por keyframe", y=1.02)
plt.savefig(OUTPUT_DIR / "05_spatial_validity.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close()

# %% [markdown]
# ## 5. ⭐ 3D — Nube de puntos del keyframe (Plotly interactivo)
#
# El plot estrella del EDA: reproyectamos el depth GT del primer frame de cada keyframe a 3D
# usando la matriz K, y coloreamos por el RGB original.
#
# $$\begin{pmatrix} x \\ y \\ z \end{pmatrix} = z \cdot K^{-1} \begin{pmatrix} u \\ v \\ 1 \end{pmatrix}$$
#
# El HTML resultante se abre en navegador y permite rotar/zoom interactivamente.

# %% Reproyección a 3D
def unproject_to_3d(depth_mm, rgb, K):
    H, W = depth_mm.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = ~np.isnan(depth_mm) & (depth_mm > 0)
    z = depth_mm[valid]
    x = (u[valid] - K[0, 2]) * z / K[0, 0]
    y = (v[valid] - K[1, 2]) * z / K[1, 1]
    color = rgb[valid] / 255.0
    return np.stack([x, y, z], axis=1), color

# generamos un point cloud por keyframe
for kf in KEYFRAMES[:3]:   # 3 para no abrumar
    K = np.load(SCARED_ROOT / kf / "calib.npz")["K1"]
    img_files = sorted((SCARED_ROOT / kf / "images").glob("*.png"))
    d_files   = sorted((SCARED_ROOT / kf / "depth").glob("*.npy"))
    if not img_files or not d_files:
        continue
    img = cv2.cvtColor(cv2.imread(str(img_files[0])), cv2.COLOR_BGR2RGB)
    depth = np.load(d_files[0])
    pts, col = unproject_to_3d(depth, img, K)

    if len(pts) > 25000:
        idx = np.random.choice(len(pts), 25000, replace=False)
        pts, col = pts[idx], col[idx]

    rgb_str = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
               for c in col]
    fig3d = go.Figure(data=[go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=rgb_str, opacity=0.85),
    )])
    fig3d.update_layout(
        title=f"Point cloud GT — {kf} (frame 0)",
        scene=dict(
            xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
    )
    out_html = OUTPUT_DIR / f"06_pointcloud_{kf}.html"
    fig3d.write_html(str(out_html))
    print(f"  ✅ Guardado {out_html}")

# %% [markdown]
# ## 6. Trayectoria 3D de la cámara
#
# Si tienes los `frame_data*.json` con poses de da Vinci, los cargas y ploteas la trayectoria.
# Esto te dice si la cámara hace barridos amplios o si está casi estática
# (cosa que importa: NeRF/Gaussian necesitan **diversidad de vista** para reconstruir bien).

# %% Trayectoria (con fallback sintético)
def load_camera_poses(kf):
    json_dir = SCARED_ROOT / kf / "frame_data"
    if not json_dir.exists():
        return None
    poses = []
    for jf in sorted(json_dir.glob("*.json")):
        try:
            with open(jf) as f:
                data = json.load(f)
            T = np.array(data["camera-pose"], dtype=np.float32).reshape(4, 4)
            poses.append(T[:3, 3])  # solo la traslación
        except Exception:
            pass
    return np.array(poses) if poses else None

trajs = {}
for kf in KEYFRAMES:
    p = load_camera_poses(kf)
    if p is None:
        # trayectoria sintética: pequeño barrido oscilante
        N = inventory.set_index("keyframe").loc[kf, "frames"]
        t = np.linspace(0, 4*np.pi, N)
        offset = hash(kf) % 100
        p = np.stack([
            offset + 8*np.cos(t) + np.random.normal(0, 0.5, N),
            5 + 4*np.sin(2*t) + np.random.normal(0, 0.3, N),
            80 + 3*np.sin(t)
        ], axis=1)
    trajs[kf] = p

fig_traj = go.Figure()
colors = ["#e63946", "#f4a261", "#2a9d8f", "#264653", "#e76f51"]
for (kf, p), c in zip(trajs.items(), colors):
    fig_traj.add_trace(go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2],
        mode="lines+markers",
        marker=dict(size=2, color=c),
        line=dict(width=3, color=c),
        name=kf,
    ))
fig_traj.update_layout(
    title="Trayectorias de cámara por keyframe",
    scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"),
    height=600, margin=dict(l=0, r=0, t=40, b=0),
)
fig_traj.write_html(str(OUTPUT_DIR / "07_camera_trajectories.html"))
print(f"✅ Guardado {OUTPUT_DIR / '07_camera_trajectories.html'}")

# %% [markdown]
# ## 7. Series temporales: depth media y luminancia por frame
#
# Permite detectar:
# - **Drift** (la cámara se aleja → depth aumenta sostenidamente)
# - **Cambios de iluminación** (si la luz parpadea)
# - **Frames problemáticos** (picos en saturación o caídas de validez)

# %% Serie temporal
fig, axes = plt.subplots(len(KEYFRAMES), 1, figsize=(13, 2.2*len(KEYFRAMES)),
                         sharex=False)
for ax, kf in zip(axes, KEYFRAMES):
    sub_d = df_depth[df_depth.keyframe == kf].sort_values("frame")
    sub_r = df_rgb[df_rgb.keyframe == kf].sort_values("frame")
    if len(sub_d) == 0:
        continue
    ax.plot(sub_d.frame, sub_d.p50, color="#1d3557", label="Depth mediana (mm)")
    ax.set_ylabel("Depth (mm)", color="#1d3557")
    ax.tick_params(axis="y", labelcolor="#1d3557")
    ax2 = ax.twinx()
    ax2.plot(sub_r.frame, sub_r.mean_lum, color="#e63946", alpha=0.75,
             label="Luminancia media")
    ax2.set_ylabel("Luminancia", color="#e63946")
    ax2.tick_params(axis="y", labelcolor="#e63946")
    ax.set_title(f"{kf}")
    ax.set_xlabel("Frame")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_timeseries.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close()

# %% [markdown]
# ## 8. Correlación: ¿saturación rompe la profundidad?
#
# Hipótesis: los specular highlights saturan el sensor → el structured light pattern
# no se ve → no hay GT depth ahí. Esto debería dar correlación negativa entre
# `% saturado` y `% válido en depth`.

# %% Scatter saturación vs validez
df_merge = df_rgb.merge(df_depth, on=["keyframe", "frame"])
plt.figure(figsize=(9, 5))
sns.scatterplot(data=df_merge, x="sat_pct", y="valid_pct",
                hue="keyframe", alpha=0.65, s=30, palette="Set2")
plt.xlabel("% pixeles saturados (RGB > 240)")
plt.ylabel("% pixeles con depth GT válido")
plt.title("Saturación vs validez de depth GT")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_saturation_vs_validity.png", dpi=120,
            bbox_inches="tight")
plt.show()
plt.close()

print("\nCorrelación de Pearson saturación↔validez por keyframe:")
for kf in KEYFRAMES:
    sub = df_merge[df_merge.keyframe == kf]
    if len(sub) > 5:
        r = sub[["sat_pct", "valid_pct"]].corr().iloc[0, 1]
        print(f"  {kf}: r = {r:+.3f}")

# %% [markdown]
# ## 9. Visualización del split train/test
#
# Recordatorio del setup: **1 de cada 8 frames es test**, los otros 7 son train.
# Visualizándolo se ve que hay buena cobertura temporal de test (no se concentra al final).

# %% Train/test split
fig, ax = plt.subplots(figsize=(14, 3.5))
y_off = 0
yticks, ylabels = [], []
for kf in KEYFRAMES:
    N = inventory.set_index("keyframe").loc[kf, "frames"]
    test_idx = np.arange(0, N, 8)
    train_idx = np.array([i for i in range(N) if i % 8 != 0])
    ax.scatter(train_idx, np.full_like(train_idx, y_off, dtype=float),
               c="#2a9d8f", s=2, alpha=0.4)
    ax.scatter(test_idx, np.full_like(test_idx, y_off, dtype=float),
               c="#e63946", s=25, marker="|")
    yticks.append(y_off); ylabels.append(kf)
    y_off += 1
ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
ax.set_xlabel("Frame index")
ax.set_title("Split train (verde) / test (rojo) — regla every-8th frame")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_train_test_split.png", dpi=120,
            bbox_inches="tight")
plt.show()
plt.close()

# %% [markdown]
# ## 10. Tabla resumen final
#
# Una sola tabla con todo lo que necesitas saber por keyframe.

# %% Resumen
summary = (df_merge.groupby("keyframe")
           .agg(frames_sampled=("frame", "count"),
                mean_lum=("mean_lum", "mean"),
                sat_pct_mean=("sat_pct", "mean"),
                depth_p50_mm=("p50", "mean"),
                depth_p5_mm=("p5", "mean"),
                depth_p95_mm=("p95", "mean"),
                valid_pct_mean=("valid_pct", "mean"))
           .round(2))
summary["range_mm"] = (summary["depth_p95_mm"] -
                       summary["depth_p5_mm"]).round(2)
print("\n📋 Resumen final por keyframe:\n")
print(summary.to_string())
summary.to_csv(OUTPUT_DIR / "11_summary.csv")

# %% [markdown]
# ## 11. Diagnóstico: ¿qué keyframe será más difícil?
#
# Ranking heurístico de dificultad para reconstrucción 3D:
# - Más specular → peor brightness consistency → más difícil para 3DGS y NeRF clásico
# - Menos validez de depth GT → menos supervisión disponible
# - Mayor rango de profundidad → más diversidad geométrica (puede ser bueno o malo)

# %% Difficulty score
diff = pd.DataFrame(index=summary.index)
diff["specular_score"] = summary["sat_pct_mean"] / summary["sat_pct_mean"].max()
diff["sparsity_score"] = 1 - summary["valid_pct_mean"] / 100
diff["range_score"]    = summary["range_mm"] / summary["range_mm"].max()
diff["difficulty"]     = (0.5*diff["specular_score"] +
                          0.3*diff["sparsity_score"] +
                          0.2*diff["range_score"])
diff = diff.sort_values("difficulty", ascending=False).round(3)
print("\n🏆 Ranking de dificultad (1 = más difícil):\n")
print(diff.to_string())

plt.figure(figsize=(9, 4.5))
diff[["specular_score", "sparsity_score", "range_score"]].plot.bar(
    stacked=True, ax=plt.gca(), colormap="Set2", edgecolor="white")
plt.title("Dificultad relativa por keyframe (composición)")
plt.ylabel("Score acumulado")
plt.xticks(rotation=0)
plt.legend(loc="upper right", bbox_to_anchor=(1.18, 1))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "12_difficulty_ranking.png", dpi=120,
            bbox_inches="tight")
plt.show()
plt.close()

# %% [markdown]
# ## ¿Y ahora qué?
#
# Con esta información puedes:
#
# 1. **Decidir el orden de experimentos**: empieza por el keyframe más fácil para depurar
#    el pipeline; luego escala al más difícil.
# 2. **Ajustar hyperparams por dificultad**: keyframes con más specular pueden necesitar
#    más iteraciones de NeRF, más densificación en EndoGaussian, o el módulo IID activado
#    en MonoIIF (que es justo para eso).
# 3. **Reportar el diagnóstico en tu thesis/paper**: este EDA justifica por qué SCARED
#    es un benchmark no trivial y qué retos plantea.
#
# Todos los outputs están en `./eda_outputs/`. Los `.html` se abren en navegador.

print("\n" + "="*60)
print("✅ EDA COMPLETO")
print("="*60)
print(f"Outputs en: {OUTPUT_DIR.resolve()}")
for p in sorted(OUTPUT_DIR.iterdir()):
    print(f"  • {p.name}")
