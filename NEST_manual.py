import pickle
import numpy as np
import random
# from novatel_math.coordinateconverter import great_circle_distance as dist
from matplotlib import pyplot as plt

NOISE_FLOOR = -83.429

TRUTH_POS = [51.4287955869, -113.8495484453, 0]

new_obs = []

def dist(p1, p2):
    # Change to radians
    phi_1 = np.radians(p1[0])
    phi_2 = np.radians(p2[0])
    lam_1 = np.radians(p1[1])
    lam_2 = np.radians(p2[1])
    del_phi = phi_2 - phi_1
    del_lam = lam_2 - lam_1

    haversine = \
        pow(np.sin(del_phi / 2.0), 2) + np.cos(phi_1) * np.cos(phi_2) * pow(np.sin(del_lam / 2.0),
                                                                            2)
    d = 6371e3 * (2.0 * np.arctan2(np.sqrt(haversine), np.sqrt(1 - haversine)))

    return float(d)

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
with open('Test1_50_smoothed_int_pos_data.txt', 'rb') as data_file:
    data = pickle.load(data_file)
# with open('DATA_TEMP.txt') as data_file:
#     for line in data_file:
#         temp = line.strip('\n').split(' ')
#         data.append([float(i) for i in temp])

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
            inside = create_inside_log(obs[2])
            if inside > 0:
                est_power = create_l(inside)  # Estimated received power with noise floor
                d = dist([obs[0], obs[1], 0], TRUTH_POS)
                a_row = create_a_row(d)
                A[row] = a_row
                Cl[row][row] = np.power(d, 0.25)
                w[row] = create_w(x_0[0], x_0[1], d, est_power)
                obs.append(d)
                new_obs.append([obs[2], d])
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

# i = 0
# delt = [0]
# nf = NOISE_FLOOR
# while not any([np.abs(v) < 0.000001for v in delt]) and not i > 10:
#     x_0_power_decay = estimate_power_decay(nf)
#     nf = estimate_nf(x_0_power_decay)
#     delt = [x_0_power_decay[0] - delt[0], x_0_power_decay[1] - delt[1], nf - delt[2]]

x_0 = estimate_power_decay(NOISE_FLOOR)

plot_model = []
p_obs = []
p_used_obs = []
x_axis = []
x_axis_all = []
diff = []
i = 0
for obs in data:
    d = dist([obs[0], obs[1], 0], TRUTH_POS)
    model_power = get_model_power(x_0[0], x_0[1], d)
    if model_power is not None:
        i += 1
        x_axis.append(d)
        p_obs.append(obs[2])
        plot_model.append(model_power)
        diff.append(np.power((model_power-obs[2]), 2))

print('RMS = {}'.format(np.sqrt(np.mean(np.array(diff)**2))))
ax1 = plt.figure().add_subplot(111)
ax2 = plt.figure().add_subplot(111)

ax1.scatter(x_axis, plot_model, s=2)
ax1.scatter(x_axis, p_obs, s=2)
ax2.scatter(x_axis, diff, s=2)
plt.show()



# Try to sim data

#Write the existing data:
sim_file = open('simulated_data.txt','w+')
for obs in new_obs:
    sim_file.write(f'{obs[1]} {obs[0]}\n')


sim_file.close()