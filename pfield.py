import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp2d, griddata, LinearNDInterpolator
from scipy.integrate import solve_ivp



f_size = np.array([
    [0, 10], #x
    [0, 10], #y
])

border = 2
obs = []
for i in range(10):
    obs.append((np.random.uniform(f_size[0, 0]+border, f_size[0, 1]-border), np.random.uniform(f_size[1, 0]+border, f_size[1, 1]-border), np.random.uniform(-10, 0.2)))
obs = np.array(obs)



def get_force(x, y, strength):
    dist = (x**2 + y**2)**0.5 + 0.5
    force = strength * 1/(dist**2)
    force_x = force * x/dist
    force_y = force * y/dist
    return (force_x, force_y)

resolution = 0.5

X, Y = np.mgrid[f_size[0, 0]:f_size[0, 1]+resolution:resolution, f_size[1, 0]:f_size[1, 1]+resolution:resolution]
forces = np.zeros(shape=(len(X), len(Y), 2))
for x_i in range(len(X)):
    for y_i in range(len(Y)):
        x = X[x_i, y_i]
        y = Y[x_i, y_i]
        for ob in obs:
            forces[x_i, y_i] += get_force(x - ob[0], y - ob[1], ob[2])

force_fieldx = LinearNDInterpolator(list(zip(X.flatten(), Y.flatten())), forces[:,:,0].flatten())
force_fieldy = LinearNDInterpolator(list(zip(X.flatten(), Y.flatten())), forces[:,:,1].flatten())


def model(t, state):
    x = state[0]
    x_d = state[1]
    y = state[2]
    y_d = state[3]

    mass = 1

    x_dd = force_fieldx(x, y) / mass
    y_dd = force_fieldy(x, y) / mass

    return [x_d, x_dd, y_d, y_dd]


sol = solve_ivp(model, y0=[2,1,2,1], t_span=(0, 20), max_step=0.2)
path_x = sol.y[0]
path_y = sol.y[2]


plt.ion()
plt.figure(figsize=(12, 8))

resolution_high = 0.1
X_high, Y_high = np.mgrid[f_size[0, 0]:f_size[0, 1]:resolution_high, f_size[1, 0]:f_size[1, 1]:resolution_high]
plt.quiver(X_high.flatten(), Y_high.flatten(), force_fieldx(X_high.flatten(), Y_high.flatten()), force_fieldy(X_high.flatten(), Y_high.flatten()))


plt.xlim(f_size[0])
plt.ylim(f_size[1])
plt.scatter(obs[:, 0], obs[:, 1], c=obs[:, 2])
plt.colorbar()

for i in range(len(path_x)):
    plt.plot(path_x[:i], path_y[:i], c='r')   
    plt.draw()
    plt.pause(0.01)


# plt.show()