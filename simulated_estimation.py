import pickle
import numpy as np
# from novatel_math.coordinateconverter import great_circle_distance as dist
from matplotlib import pyplot as plt

NOISE_FLOOR = -83.429

TRUTH_POS = [51.4287955869, -113.8495484453, 0]

new_obs = []


def create_inside_log(p):

    term1 = np.power(10, p/10)
    term2 = np.power(10, NOISE_FLOOR/10)
    return term1 - term2

def create_l(inside):
    return 10*np.log10(inside)

def create_a_row(d):
    return [-1, np.log10(d)]

def creat_plot_data(x1, x2, d, p):
    return (-1*x1 + x2*np.log10(d), p)

def create_w(x1, x2, d, p):
    return -1*x1 + x2*np.log10(d) - p

# Read in data
data = []
with open('simulated_data.txt', 'r+') as data_file:
    data_str = data_file.readlines()
    for line in data_str:
        data.append([float(d) for d in line.split()])

def get_model_power(x1, x2, d):
    # inside = np.power(10, (x1 - x2*np.log10(d)/10)) + np.power(10, NOISE_FLOOR/10)
    # return 10*np.log10(inside)
    inside = 10**((x2 * np.log10(d) - x1) / 10.0) + 10**(NOISE_FLOOR / 10.0)

    return 10 * np.log10(inside) if inside > 0 else None

def estimate_power_decay(noise_floor):

    # Create Design matrix
    x_0 = [-14, -7]
    i = 0
    delt = [10]
    while not any([np.abs(v) < 0.000001for v in delt]) and not i > 10:
        i += 1
        print(i)
        A = np.empty([len(data), 2])
        w = np.empty(len(data))
        Cl = np.zeros([len(data), len(data)])
        row = 0
        used_data = []
        for obs in data:
            inside = create_inside_log(obs[1])
            if inside > 0:
                est_power = create_l(inside)  # Estimated received power with noise floor
                d = obs[0]
                a_row = create_a_row(d)
                A[row] = a_row
                Cl[row][row] = np.power(d, 0.25)
                w[row] = create_w(x_0[0], x_0[1], d, est_power)
                used_data.append(obs)
            else:
                w[row] = 0
                A[row] = [0, 0]
            row += 1
        at_cl = np.matmul(A.transpose(), Cl)
        N = np.matmul(at_cl, A)

        U = np.matmul(at_cl, w)
        delt = np.matmul(np.linalg.inv(N), U)
        x_0 = [x_0[0]-delt[0], x_0[1]-delt[1]]
        with open('used_data.txt', 'w+') as used_data_file:
            for data_out in used_data:
                [used_data_file.write(f'{dat}\t') for dat in data_out]
                used_data_file.write('\n')
    return x_0

x_0 = estimate_power_decay(NOISE_FLOOR)

plot_model = []
p_obs = []
p_used_obs = []
x_axis = []
x_axis_all = []
diff = []
i = 0
for obs in data:
    d = obs[0]
    model_power = get_model_power(x_0[0], x_0[1], d)
    if model_power is not None:
        i += 1
        x_axis.append(d)
        p_obs.append(obs[1])
        plot_model.append(model_power)
        diff.append(np.power((model_power-obs[1]), 2))

print('RMS = {}'.format(np.sqrt(np.mean(np.array(diff)**2))))
ax1 = plt.figure().add_subplot(111)
ax2 = plt.figure().add_subplot(111)

ax1.scatter(x_axis, plot_model, s=2)
ax1.scatter(x_axis, p_obs, s=2)
ax2.scatter(x_axis, diff, s=2)
plt.show()