"""
PROYECTO FIELD LOOP - PARTE 1: malla, condición inicial y campo magnético.

Reproduce en Python la condición inicial del test Field_Loop de PLUTO
(Test_Problems/MHD/Field_Loop, init.c) y la compara con la salida t = 0 de PLUTO.

Física (init.c de PLUTO):
  - rho = 1, p = 1, v = (2, 1)  [V0 = sqrt(5)],  vz = Bz = 0
  - Potencial vectorial:  A_z = A0 (R - r)  si r <= R ;  0 si r > R
    con A0 = 1e-3, R = 0.3, r = sqrt(x^2 + y^2)
  - Campo magnético:  B = rot(A_z z) ->  Bx = dA_z/dy ,  By = -dA_z/dx
Dominio: x en [-1, 1], y en [-0.5, 0.5], periódico.  Malla 128 x 64.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Parámetros físicos y de malla (iguales a PLUTO, configuración 05)
# ---------------------------------------------------------------------------
RHO0, P0 = 1.0, 1.0            # densidad y presión
VX, VY = 2.0, 1.0              # velocidad del flujo, V0 = sqrt(5)
A0, R = 1.0e-3, 0.3            # amplitud del potencial y radio del lazo
XMIN, XMAX = -1.0, 1.0
YMIN, YMAX = -0.5, 0.5


def malla(nx, ny):
    """Devuelve centros de celda (xc, yc), esquinas (xn, yn) y tamaños dx, dy.

    PLUTO guarda las variables en los CENTROS de celda; el potencial A_z
    (Constrained Transport) vive en las ESQUINAS de las celdas.
    """
    dx = (XMAX - XMIN) / nx
    dy = (YMAX - YMIN) / ny
    xc = XMIN + (np.arange(nx) + 0.5) * dx     # centros
    yc = YMIN + (np.arange(ny) + 0.5) * dy
    xn = XMIN + np.arange(nx + 1) * dx         # esquinas (incluye ambos bordes)
    yn = YMIN + np.arange(ny + 1) * dy
    return xc, yc, xn, yn, dx, dy


def potencial_Az(x, y):
    """A_z = A0 (R - r) dentro del lazo, 0 fuera (init.c de PLUTO)."""
    r = np.sqrt(x**2 + y**2)
    return np.where(r <= R, A0 * (R - r), 0.0)


def campo_desde_Az(Az, dx, dy):
    """B = rot(A_z z) con diferencias finitas sobre la malla escalonada.

    Az tiene forma (ny+1, nx+1) y está en las esquinas.
      Bx en las caras x (ny, nx+1):  Bx = [A_z(j+1/2) - A_z(j-1/2)] / dy
      By en las caras y (ny+1, nx):  By = -[A_z(i+1/2) - A_z(i-1/2)] / dx
    Luego se promedian las dos caras de cada celda para obtener el valor
    en el centro (así lo hace PLUTO con Constrained Transport).
    Este procedimiento garantiza div(B) = 0 exactamente en forma discreta.
    """
    Bx_caras = (Az[1:, :] - Az[:-1, :]) / dy
    By_caras = -(Az[:, 1:] - Az[:, :-1]) / dx
    Bx = 0.5 * (Bx_caras[:, 1:] + Bx_caras[:, :-1])   # centros (ny, nx)
    By = 0.5 * (By_caras[1:, :] + By_caras[:-1, :])
    divB = ((Bx_caras[:, 1:] - Bx_caras[:, :-1]) / dx
            + (By_caras[1:, :] - By_caras[:-1, :]) / dy)
    return Bx, By, divB


def condicion_inicial(nx, ny):
    """Todas las variables en t = 0, en los centros de celda, forma (ny, nx)."""
    xc, yc, xn, yn, dx, dy = malla(nx, ny)
    XN, YN = np.meshgrid(xn, yn)                 # esquinas
    Az = potencial_Az(XN, YN)
    Bx, By, divB = campo_desde_Az(Az, dx, dy)
    forma = (ny, nx)
    estado = {
        "rho": np.full(forma, RHO0), "prs": np.full(forma, P0),
        "vx1": np.full(forma, VX),   "vx2": np.full(forma, VY),
        "vx3": np.zeros(forma),
        "Bx1": Bx, "Bx2": By, "Bx3": np.zeros(forma),
    }
    return xc, yc, Az, estado, divB


if __name__ == "__main__":
    nx, ny = 128, 64
    xc, yc, Az, U, divB = condicion_inicial(nx, ny)
    print(f"Malla {nx}x{ny}: dx = {xc[1]-xc[0]:.6f}, dy = {yc[1]-yc[0]:.6f}")
    print(f"max|Bx| = {abs(U['Bx1']).max():.4e},  max|By| = {abs(U['Bx2']).max():.4e}")
    print(f"max|div B| (Python) = {abs(divB).max():.2e}")
    beta = P0 / (0.5 * (U['Bx1']**2 + U['Bx2']**2).max())
    print(f"beta minimo del plasma = p / (B^2/2) = {beta:.2e}")

    # ----------------- Comparación con PLUTO en t = 0 ------------------
    try:
        import pyPLUTO as pp
        D = pp.Load(nout=0, path="./", text=False)
        # PyPLUTO entrega arreglos con forma (nx, ny): se trasponen a (ny, nx)
        dBx = U["Bx1"] - np.asarray(D.Bx1).T
        dBy = U["Bx2"] - np.asarray(D.Bx2).T
        print(f"Diferencia con PLUTO en t=0: max|dBx| = {abs(dBx).max():.2e}, "
              f"max|dBy| = {abs(dBy).max():.2e}")
    except Exception as e:
        print("No se pudo cargar PLUTO (corre el script desde la carpeta de la simulación):", e)

    # ----------------- Figura de la condición inicial -------------------
    fig, axs = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for ax, q, tit in zip(axs, [Az * 1e3, U["Bx1"] * 1e3, U["Bx2"] * 1e3],
                          [r"$A_z\ [\times10^{-3}]$", r"$B_x\ [\times10^{-3}]$", r"$B_y\ [\times10^{-3}]$"]):
        ext = [XMIN, XMAX, YMIN, YMAX]
        im = ax.imshow(q, origin="lower", extent=ext, cmap="RdBu_r" if "B" in tit else "viridis")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(tit); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle("Field Loop - condición inicial (Python, 128x64)")
    fig.savefig("python_condicion_inicial.png", dpi=150)
    print("Guardada: python_condicion_inicial.png")
