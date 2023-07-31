import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append('..')

from src import NODE_solve_Lorenz as sol 
from src import NODE_util as util

def plot_attractor(optim_name, num_epoch, lr, time_step):
    ''' func: plotting the attractor '''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)

    ##### create data for train, test, and extrapolation #####
    if time_step == 5e-4:
        transition_phase = 60000
        X, Y, X_test, Y_test = sol.create_data(0, 40, torch.Tensor([ -8., 7., 27.]), 80001, n_train=18000, n_test=2000, n_nodes=3, n_trans=transition_phase) # 22.5%

        true_traj = sol.simulate(0, 80, torch.Tensor([ -8., 7., 27.]), 160001)
        true_traj = true_traj[transition_phase:]

    elif time_step == 5e-3:
        transition_phase = 6000
        X, Y, X_test, Y_test = sol.create_data(0, 40, torch.Tensor([ -8., 7., 27.]), 8001, n_train=1800, n_test=200, n_nodes=3, n_trans=transition_phase)

        true_traj = sol.simulate(0, 80, torch.Tensor([ -8., 7., 27.]), 16001)
        true_traj = true_traj[transition_phase:]

    elif time_step == 1e-2:
        transition_phase = 3000
        X, Y, X_test, Y_test = sol.create_data(0, 40, torch.Tensor([ -8., 7., 27.]), 4001, n_train=900, n_test=100, n_nodes=3, n_trans=transition_phase)

        true_traj = sol.simulate(0, 80, torch.Tensor([ -8., 7., 27.]), 8001)
        true_traj = true_traj[transition_phase:]



    print("testing initial point: ", true_traj[0])
    print("created data!")

    ##### plot training data trajectory #####
    util.plot_traj_lorenz(X, optim_name, time_step)

    ##### create model #####
    m = sol.create_NODE(device, n_nodes=3, T=time_step)
    print("created model!")

    ##### train #####
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay =5e-4)

    pred_train, true_train, pred_test, loss_hist, test_loss_hist = sol.train(m,
                                                                             device,
                                                                             X,
                                                                             Y,
                                                                             X_test,
                                                                             Y_test, 
                                                                             true_traj,
                                                                             optimizer,
                                                                             criterion,
                                                                             epochs=num_epoch,
                                                                             lr=lr,
                                                                             time_step=time_step)
    print("train loss: ", loss_hist[-1])
    #print("test loss: ", test_loss_hist[-1])

    ##### Save Training Loss #####
    loss_csv = np.asarray(loss_hist)
    np.savetxt('expt_lorenz/'+ optim_name + '/' + str(time_step) + '/' +"training_loss.csv", loss_csv, delimiter=",")

    ##### Plot Phase Space #####
    util.plot_phase_space_lorenz(pred_test, Y_test, optim_name, lr, time_step)

    ##### Plot Time Space #####
    util.plot_time_space_lorenz(X, X_test, Y_test, pred_train, true_train, pred_test, loss_hist, optim_name, lr, num_epoch, time_step)

    return 


##### run experiment #####    
plot_attractor('AdamW', 12000, 5e-4, 1e-2) # optimizer name, epoch, lr, time_step