import pickle
import numpy as np
# from novatel_math.coordinateconverter import great_circle_distance as dist
from matplotlib import pyplot as plt
import random

NOISE_FLOOR = -77.73

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


def get_model_power(x1, x2, d):
    # inside = np.power(10, (x1 - x2*np.log10(d)/10)) + np.power(10, NOISE_FLOOR/10)
    # return 10*np.log10(inside)
    inside = 10**((x2 * np.log10(d) - x1) / 10.0) + 10**(NOISE_FLOOR / 10.0)

    return 10 * np.log10(inside) if inside > 0 else None

def EST_estimate_power_decay(data):

    x_0 = [-14, -7]
    model_p = []
    A = np.empty([len(data), 2])
    w = np.empty(len(data))
    row = 0
    for obs in data:
        logd = np.log10(obs[0])
        model_power = x_0[0] + x_0[1]*logd
        model_p.append(model_power)
        A[row] = [1, logd]
        w[row] = model_power - obs[1]
        row += 1

    At = np.transpose(A)
    N = np.matmul(At, A)
    U = np.matmul(At, w)
    delt = np.matmul(np.linalg.inv(N), U)
    x_0 = [x_0[0] - delt[0], x_0[1] - delt[1]]

    return x_0


def NEST_estimate_power_decay(data):

    # Create Design matrix
    x_0 = [-14, -7]
    i = 0
    delt = [10]
    while not any([np.abs(v) < 0.000001for v in delt]) and not i > 10:
        i += 1
        # print(i)
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
                if d > 100:
                    weight = 1/np.power(d, 2)
                else:
                    weight = 1

                Cl[row][row] = weight#;np.power(d, 0.5)
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



# Read in data
data_real = []
with open('spot2_data_power_distance.dat', 'r+') as data_file:
    data_str = data_file.readlines()
    for line in data_str:
        data_real.append([float(d) for d in line.split()])


rms_res_NEST = []
rms_res_EST = []

for i in range(0, 100000, 10000):
    print(f"Starting {i}")
    data = data_real.copy()
    start_dist = 800
    power = NOISE_FLOOR
    new_sim_data = []
    for j in range(start_dist, start_dist+i, 10):
        d = j
        p = NOISE_FLOOR + (0.1*random.random())
        data.append([d, p])

    x_0_NEST = NEST_estimate_power_decay(data)
    x_0_EST = EST_estimate_power_decay(data)

    plot_model_NEST = []
    plot_model_EST = []
    p_obs_NEST = []
    p_obs_EST = []
    p_used_obs = []
    x_axis = []
    x_axis_all = []
    diff_NEST = []
    diff_EST = []
    i = 0
    j=0
    for obs in data:
        d = obs[0]
        NEST_model_power = get_model_power(x_0_NEST[0], x_0_NEST[1], d)
        EST_model_power = x_0_EST[0] + x_0_EST[1]*np.log10(d)
        if NEST_model_power is not None:
            i += 1
            x_axis.append(d)
            p_obs_NEST.append(obs[1])
            plot_model_NEST.append(NEST_model_power)
            diff_NEST.append(np.power((NEST_model_power-obs[1]), 2))
        if EST_model_power is not None:
            j += 1
            p_obs_EST.append(obs[1])
            plot_model_EST.append(EST_model_power)
            diff_EST.append(np.power((EST_model_power-obs[1]), 2))


    rms_NEST = np.sqrt(np.mean(np.array(diff_NEST) ** 2))
    rms_res_NEST.append(rms_NEST)
    rms_EST = np.sqrt(np.mean(np.array(diff_EST) ** 2))
    rms_res_EST.append(rms_EST)


print(np.mean(p_obs_NEST))
print(NOISE_FLOOR)

#plt.plot(rms_res)
ax1 = plt.figure().add_subplot(111)
ax1.scatter([d for d in range(800, 800+len(rms_res_NEST))], rms_res_NEST, s=2)
ax1.scatter([d for d in range(800, 800+len(rms_res_EST))], rms_res_EST, s=2)
ax1.legend(['NEST', 'Est'])
plt.show()

print('RMS NEST = {}'.format(np.sqrt(np.mean(np.array(diff_NEST)**2))))
print('RMS EST = {}'.format(np.sqrt(np.mean(np.array(diff_EST)**2))))
ax1 = plt.figure().add_subplot(111)
ax2 = plt.figure().add_subplot(111)
ax3 = plt.figure().add_subplot(111)
ax4 = plt.figure().add_subplot(111)

ax1.scatter(x_axis, plot_model_NEST, s=2)
ax1.scatter(x_axis, p_obs_NEST, s=2)
ax1.set_title('NEST Model')
ax2.scatter(x_axis, diff_NEST, s=2)
ax2.set_title('NEST Diff')
ax3.scatter(x_axis, plot_model_EST, s=2)
ax3.scatter(x_axis, p_obs_EST, s=2)
ax3.set_title('EST Model')
ax4.scatter(x_axis, diff_EST, s=2)
ax4.set_title('EST Diff')
plt.show()