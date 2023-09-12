import torch
import numpy as np
from matplotlib.pyplot import * 
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
        # transition time is decided so that Model Time Unit (delta t * t_n) = 1
        tran = 2000 
        X, Y, X_test, Y_test = sol.create_data(0, 120, 
                                torch.Tensor([ -8., 7., 27.]), 120*2000+1, 
                                n_train=10000, n_test=1800, n_nodes=3, n_trans=tran)

        # integration time length is decided to make real time length equal to 2.5
        true_traj = sol.simulate(0, t_n, 
                                torch.Tensor([ 0.1, 0.1, 0.1]), t_n*2000+1)
        true_traj = true_traj[tran:]
        #true_traj = torch.tensor([[0,0,0],[1,2,3]])

        # #--- [0,100] ---#
        # t_n = 2000
        # X, Y, X_test, Y_test = sol.create_data(0, 100, torch.Tensor([ -8., 7., 27.]), 200001, n_train=64000, n_test=14000, n_nodes=3, n_trans=2000)

        # true_traj = sol.simulate(0, t_n, torch.Tensor([ -8., 7., 27.]), t_n*2000+1)

    elif time_step == 5e-3:

        t_n = 200
        tran = 200
        X, Y, X_test, Y_test = sol.create_data(0, 120, 
                                torch.Tensor([ -8., 7., 27.]), 120*200+1,
                                n_train=10000, n_test=1800, n_nodes=3, n_trans=tran)

        true_traj = sol.simulate(0, t_n, 
                                torch.Tensor([ 0.1, 0.1, 0.1]), t_n*200+1)
        true_traj = true_traj[tran:]

    elif time_step == 1e-2:

        t_n = 100
        tran = 100
        X, Y, X_test, Y_test = sol.create_data(0, 120, 
                                torch.Tensor([ -8., 7., 27.]), 12001, 
                                n_train=10000, n_test=1800, n_nodes=3, n_trans=tran)
        # test multi-time step with new initial points
        true_traj = sol.simulate(0, t_n, 
                                torch.Tensor([ 0.1, 0.1, 0.1]), t_n*100 + 1)
        true_traj = true_traj[tran:]


    print("testing initial point: ", true_traj[0])
    print("created data!")

    ##### plot training data trajectory #####
    util.plot_traj_lorenz(X, optim_name, time_step, False)

    ##### create model #####
    m = sol.create_NODE(device, n_nodes=3, T=time_step)
    torch.cuda.empty_cache()
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

    ##### Save Model #####
    path = "expt_lorenz/"+optim_name+"/"+str(time_step)+'/'+'model.pt'
    torch.save(m.state_dict(), path)
    
    ##### Save True Trajectory #####
    true_traj_csv = np.asarray(true_traj)
    np.savetxt('expt_lorenz/'+ optim_name + '/' + str(time_step) + '/' +"true_traj.csv", true_traj_csv, delimiter=",")

    ##### Save Training/Test Loss #####
    loss_csv = np.asarray(loss_hist)
    test_loss_csv = np.asarray(test_loss_hist)
    np.savetxt('expt_lorenz/'+ optim_name + '/' + str(time_step) + '/' +"training_loss.csv", loss_csv, delimiter=",")
    np.savetxt('expt_lorenz/'+ optim_name + '/' + str(time_step) + '/' +"test_loss.csv", test_loss_csv, delimiter=",")

    ##### Plot Phase Space #####
    util.plot_phase_space_lorenz(pred_test, Y_test, optim_name, lr, time_step, False)

    ##### Plot Time Space #####
    util.plot_time_space_lorenz(X, X_test, Y_test, pred_train, true_train, pred_test, loss_hist, optim_name, lr, num_epoch, time_step, False)

    return 


##### run experiment #####    
if __name__ == '__main__':
    plot_attractor('AdamW', 8000, 5e-4, 1e-2) # optimizer name, epoch, lr, time_step