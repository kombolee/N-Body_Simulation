import numpy as np
import taichi as ti

paused = np.zeros(())

# gravitational constant 6.67408e-11, using 1 for simplicity
G = 1
# number of planets
N = 600
# unit mass
m = 1
# galaxy size
galaxy_size = 0.2
# planet radius (for rendering)
planet_radius = 6
# init vel
init_vel = 120

# time-step size
h = 1e-4
# substepping
substepping = 10

# center of the screen
center = np.zeros(2)

# pos, vel and force of the planets
# Nx2 vectors

pos = np.zeros((N,2))
vel = np.zeros((N,2))
force = np.zeros((N,2))

def initialize():
    center = [0.5, 0.5]
    for i in range(N):
        theta = np.random.rand() * 2 * np.pi
        r = (np.sqrt(np.random.rand()) * 0.6 + 0.4) * galaxy_size
        offset = r * np.array([np.cos(theta), np.sin(theta)])
        pos[i] = center + offset
        vel[i] = [-offset[1], offset[0]]
        vel[i] *= init_vel


def compute_force():

    # clear force
    for i in range(N):
        force[i] = [0.0, 0.0]

    # compute gravitational force
    for i in range(N):
        p = pos[i]
        for j in range(N):
            if i != j:  # double the computation for a better memory footprint and load balance
                diff = p - pos[j]
                #r = diff.norm(1e-5)
                r = np.linalg.norm(diff)
                # gravitational force -(GMm / r^2) * (diff/r) for i
                f = -G * m * m * (1.0 / r)**3 * diff
                # assign to each particle
                force[i] += f


def update():
    dt = h / substepping
    for i in range(N):
        #symplectic euler
        vel[i] += dt * force[i] / m
        pos[i] += dt * vel[i]


gui = ti.GUI('N-body problem', (800, 800))

initialize()
while gui.running:

    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
        #if e.key == 'e':
            exit()
        elif e.key == 'r':
            initialize()
        elif e.key == ti.GUI.SPACE:
            paused[None] = not paused[None]

    if not paused[None]:
        for i in range(substepping):
            compute_force()
            update()

    gui.circles(pos, color=0xffffff, radius=planet_radius)
    gui.show()
