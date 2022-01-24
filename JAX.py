import numpy as onp
import jax.numpy as jnp
import taichi as ti

paused = jnp.zeros(())

# gravitational constant 6.67408e-11, using 1 for simplicity
G = 1
# number of planets
N = 300
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
substepping = 2

# center of the screen
center = jnp.zeros(2)

# pos, vel and force of the planets
# Nx2 vectors


pos = jnp.zeros((N,2))
vel = jnp.zeros((N,2))
force = jnp.zeros((N,2))

def initialize():
    center = [0.5, 0.5]

    for i in range(N):
        theta = onp.random.rand() * 2 * onp.pi
        r = (jnp.sqrt(onp.random.rand()) * 0.6 + 0.4) * galaxy_size

        offset = r * jnp.array([jnp.cos(theta), jnp.sin(theta)])
        offset = offset.tolist()
        position = jnp.add(jnp.asarray(center), jnp.asarray(offset))

        global pos
        global vel
        pos = pos.at[i].set(position)

        vel = vel.at[i].set([-offset[1], offset[0]])
        vel = vel.at[i].set(vel[i] * init_vel)


def compute_force():

    # clear force
    for i in range(N):
        global force
        force = force.at[i].set([0.0, 0.0])

    # compute gravitational force
    for i in range(N):
        p = pos[i]
        for j in range(N):
            if i != j:  # double the computation for a better memory footprint and load balance
                diff = p - pos[j]
                #r = diff.norm(1e-5)
                r = onp.linalg.norm(diff)
                # gravitational force -(GMm / r^2) * (diff/r) for i
                f = -G * m * m * (1.0 / r)**3 * diff
                # assign to each particle
                force = force.at[i].set(force[i] + f)


def update():
    dt = h / substepping
    for i in range(N):
        #symplectic euler
        global vel
        global pos
        vel = vel.at[i].set((vel[i] + (dt * force[i] / m)))

        pos = pos.at[i].set((pos[i] + (dt * vel[i])))


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

