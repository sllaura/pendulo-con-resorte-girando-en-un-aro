import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------------------------------------
# Parámetros del sistema
# -----------------------------------------------------------
g = 9.81
m = 1.0       # masa del péndulo
k = 60.0      # constante elástica del resorte
r0 = 1.0      # longitud natural del resorte
a = 0.6       # radio del aro sobre el que gira el soporte
omega = 2.0   # velocidad angular de rotación del soporte (rad/s)

# -----------------------------------------------------------
# Condiciones iniciales
# -----------------------------------------------------------
r_0 = 1.2                  # longitud inicial del resorte
dr_0 = 0.0                 # velocidad radial inicial
th_0 = np.radians(30.0)    # ángulo inicial del péndulo
dth_0 = 0.0                # velocidad angular inicial del péndulo

# -----------------------------------------------------------
# Parámetros del método numérico
# -----------------------------------------------------------
tmax = 20
dt = 0.005
STRIDE = 3

# -----------------------------------------------------------
# Dinámica: péndulo-resorte cuyo soporte gira sobre un aro
# de radio "a" con velocidad angular constante "omega".
#
# Ecuaciones obtenidas por Euler-Lagrange (ver notebook):
#   ddr  = [-k(r-r0) + m r dθ² + m g cosθ + m a ω² sin(θ-ωt)] / m
#   ddθ  = [-2 dr dθ + a ω² cos(θ-ωt) - g sinθ] / r
# -----------------------------------------------------------
def dyn(t, y):
    r, dr, th, dth = y
    fase = th - omega * t

    ddr = (-k * (r - r0) + m * r * dth**2
           + m * g * cos(th) + m * a * omega**2 * sin(fase)) / m

    ddth = (-2 * dr * dth + a * omega**2 * cos(fase) - g * sin(th)) / r

    return np.array([dr, ddr, dth, ddth])

# -----------------------------------------------------------
# Integrador RK4
# -----------------------------------------------------------
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# -----------------------------------------------------------
# Integración
# -----------------------------------------------------------
n = int(tmax / dt)
t = np.linspace(0, n*dt, n+1)
y = np.empty((n+1, 4))
y[0] = np.array([r_0, dr_0, th_0, dth_0])
for i in range(n):
    y[i+1] = rk4(dyn, t[i], y[i], dt)

r, dr, th, dth = y[:, 0], y[:, 1], y[:, 2], y[:, 3]

# -----------------------------------------------------------
# Cinemática: posición del soporte (sobre el aro) y del bob
# -----------------------------------------------------------
x_pivot = a * cos(omega * t)
y_pivot = a * sin(omega * t)
x_bob = x_pivot + r * sin(th)
y_bob = y_pivot - r * cos(th)

# -----------------------------------------------------------
# Figura (colores del péndulo con soporte deslizante y resorte)
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
R = a + r.max() + 0.4
ax.set_xlim(-R, R); ax.set_ylim(-R, R)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Péndulo-resorte con soporte girando en un aro")
ax.tick_params(axis="both", labelsize=12)
ax.grid(alpha=0.3)

# Aro fijo (guía circular sobre la que se desplaza el soporte)
hoop = plt.Circle((0, 0), a, fill=False, lw=1.5, color="black", ls="--", alpha=0.6)
ax.add_patch(hoop)

pivot, = ax.plot([], [], "o", ms=10, color="navy", mec="black", mew=1.2, zorder=6)
bob, = ax.plot([], [], "o", ms=16, color="blue", mec="black", mew=1.2, zorder=7)
spring_line, = ax.plot([], [], lw=1.5, color="black", zorder=4)
# Rastro (trace) del movimiento del bob. Se dibuja en un color distinto
# (rojo, tenue) para no confundirse con el resorte ni con el bob.
trace, = ax.plot([], [], "-", lw=1, alpha=0.5, color="blue", zorder=3)
clock = ax.text(0.05, 0.93, "", transform=ax.transAxes, fontsize=12, color="black")

l0_ref = r0  # longitud de referencia para escalar la amplitud del resorte

def spring_coords(x1, y1, x2, y2, n_coils=12):
    """Resorte helicoidal (forma de bobina, no en zigzag/'W')."""
    dx = x2 - x1
    dy = y2 - y1
    L = np.hypot(dx, dy) + 1e-9
    s_param = np.linspace(0, 1, 300)
    s = 0.5 * (1 - np.cos(np.pi * s_param))   # distribución no uniforme
    nx = -dy / L
    ny = dx / L
    amp = np.clip(0.08 * np.sqrt(l0_ref / L), 0.04, 0.10) if L > 0 else 0.08
    wave = amp * np.sin(2 * np.pi * n_coils * s)
    xs = x1 + dx * s + nx * wave
    ys = y1 + dy * s + ny * wave
    return xs, ys

def animate(i):
    pivot.set_data([x_pivot[i]], [y_pivot[i]])
    bob.set_data([x_bob[i]], [y_bob[i]])
    xs, ys = spring_coords(x_pivot[i], y_pivot[i], x_bob[i], y_bob[i])
    spring_line.set_data(xs, ys)
    trace.set_data(x_bob[:i+1], y_bob[:i+1])
    clock.set_text(f"t = {t[i]:.1f} s")
    return pivot, bob, spring_line, trace, clock

ani = FuncAnimation(fig, animate, frames=range(0, n+1, STRIDE),
                     interval=STRIDE*dt*1000, blit=True)
plt.tight_layout()
plt.show()