import torch
import numpy as np
from matplotlib.pyplot import * 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append('..')

from src import NODE_solve_Lorenz as sol 
from src import NODE_util as util

def plot_attractor(x0, x_multi_0, optim_name, num_epoch, lr, time_step):
    ''' func: plotting the attractor '''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)


    ##### create data for train, test, and extrapolation #####
    if time_step == 5e-4:
        # transition time is decided so that Model Time Unit (delta t * t_n) = 1
        # integration time length is decided to make real time length equal to 1
        iters = 5*(10**5)
        tran_n = 2000

    elif time_step == 5e-3:
        iters = 2*(10**5)
        tran_n = 200

    elif time_step == 1e-2:
        iters = 2*(10**4)
        tran_n = 100

    real_time = iters * time_step
    traj = sol.simulate(0, 120, x0, time_step)
    dataset = sol.create_data(traj, n_train=10000, n_test=1800, n_nodes=3, n_trans=tran_n)
    longer_traj = sol.simulate(0, real_time, x_multi_0, time_step)
    multistep_traj = longer_traj[tran_n:]

    print("testing initial point: ", multistep_traj[0])
    print("real time: ", real_time)
    print("created data!")

    ##### create model #####
    m = sol.create_NODE(device, n_nodes=3, n_hidden=64,T=time_step)
    torch.cuda.empty_cache()
    print("created model!")

    ##### train #####
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay =5e-4)

    pred_train, true_train, pred_test, loss_hist, test_loss_hist = sol.train(m,
                                                                             device,
                                                                             dataset, 
                                                                             multistep_traj,
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
    true_traj_csv = np.asarray(multistep_traj)
    np.savetxt('expt_lorenz/'+ optim_name + '/' + str(time_step) + '/' +"true_traj.csv", true_traj_csv, delimiter=",")

    ##### Save Training/Test Loss #####
    loss_csv = np.asarray(loss_hist)
    test_loss_csv = np.asarray(test_loss_hist)
    np.savetxt('expt_lorenz/'+ optim_name + '/' + str(time_step) + '/' +"training_loss.csv", loss_csv, delimiter=",")
    np.savetxt('expt_lorenz/'+ optim_name + '/' + str(time_step) + '/' +"test_loss.csv", test_loss_csv, delimiter=",")

    ##### Plot Phase Space #####
    X, Y, X_test, Y_test = dataset
    util.plot_phase_space_lorenz(pred_test, Y_test, optim_name, lr, time_step, False)

    ##### Plot training data trajectory #####
    util.plot_traj_lorenz(X, optim_name, time_step, False)

    ##### Plot Time Space #####
    util.plot_time_space_lorenz(X, X_test, Y_test, pred_train, true_train, pred_test, loss_hist, optim_name, lr, num_epoch, time_step, False)

    return 


##### run experiment #####    
if __name__ == '__main__':
    x0 = torch.randn(3) #torch.Tensor([ -8., 7., 27.])
    x_multi_0 = torch.randn(3) # torch.Tensor([ 0.1, 0.1, 0.1])
    plot_attractor(x0, x_multi_0, 'AdamW', 12000, 5e-4, 1e-2) # optimizer name, epoch, lr, time_step