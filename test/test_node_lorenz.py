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

        #--- [0,40] ---#
        t_n = 2000
        X, Y, X_test, Y_test = sol.create_data(0, 40, torch.Tensor([ -8., 7., 27.]), 80001, n_train=32000, n_test=8000, n_nodes=3, n_trans=0)

        # integration time length is decided to make real time length equal to 1
        true_traj = sol.simulate(0, t_n, torch.Tensor([ -8., 7., 27.]), t_n*2000+1)

        # #--- [0,100] ---#
        # t_n = 2000
        # X, Y, X_test, Y_test = sol.create_data(0, 100, torch.Tensor([ -8., 7., 27.]), 200001, n_train=64000, n_test=16000, n_nodes=3, n_trans=0)

        # true_traj = sol.simulate(0, t_n, torch.Tensor([ -8., 7., 27.]), t_n*2000+1)

    elif time_step == 5e-3:

        t_n = 200
        X, Y, X_test, Y_test = sol.create_data(0, 40, torch.Tensor([ -8., 7., 27.]), 8001, n_train=6400, n_test=1600, n_nodes=3, n_trans=0)

        true_traj = sol.simulate(0, t_n, torch.Tensor([ -8., 7., 27.]), t_n*200+1)

    elif time_step == 1e-2:

        t_n = 100
        X, Y, X_test, Y_test = sol.create_data(0, 40, torch.Tensor([ -8., 7., 27.]), 4001, n_train=3200, n_test=800, n_nodes=3, n_trans=0)

        true_traj = sol.simulate(0, t_n, torch.Tensor([ -8., 7., 27.]), t_n*100+1)
        



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
                                                                             time_step=time_step,
                                                                             integration_time=t_n)
    print("train loss: ", loss_hist[-1])
    print("test loss: ", test_loss_hist[-1])

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