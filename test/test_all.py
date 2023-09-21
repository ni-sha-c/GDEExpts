from test_node_lorenz import *
from test_metrics import *
import sys
sys.path.append('..')
from src.NODE_solve_Lorenz import *

##### run experiment #####    
def main():
# return LEs : truth, model
# input: true_model(\example), dimension of state
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)

    # Assign Initial Point of Orbit
    # 3 -> d
    x0 = torch.randn(3)
    x_multi_0 = torch.randn(3)

    # Initialize Model Parameters
    time_step = 1e-2
    lr = 5e-4
    weight_decay = 5e-4
    num_epoch = 20000
    optim_name = "AdamW"
    criterion = torch.nn.MSELoss()

    # Intialize Dataset Parameters
    tran_state = 100
    iters = 2*(10**4)
    real_time = iters * time_step
    print("real time: ", real_time)

    # Generate Training/Test/Multi-Step Prediction Data
    traj = sol.simulate(0, 180, x0, time_step)
    dataset = sol.create_data(traj, n_train=10000, n_test=7500, n_nodes=3, n_trans=tran_state)
    longer_traj = sol.simulate(0, real_time, x_multi_0, time_step)
    multistep_traj = longer_traj[tran_state:]

    # Create model
    m = sol.create_NODE(device, n_nodes=3, n_hidden=64,T=time_step).double()
    optimizer = sol.define_optimizer(optim_name, m, lr, weight_decay)

    # Train model
    pred_train, true_train, pred_test, loss_hist, test_loss_hist = sol.train(m,
                                                                             device,
                                                                             dataset, 
                                                                             multistep_traj,
                                                                             optimizer,
                                                                             criterion,
                                                                             epochs=num_epoch,
                                                                             lr=lr,
                                                                             time_step=time_step,
                                                                             integration_time=real_time)
    print("train loss: ", loss_hist[-1])
    print("test loss: ", test_loss_hist[-1])

    # Save Trained Model
    path = "expt_lorenz/"+optim_name+"/"+str(time_step)+'/'+'model.pt'
    torch.save(m.state_dict(), path)

    # Compute Jacobian Matrix and Lyapunov Exponent of Neural ODE
    LE_NODE = lyap_exps(True, longer_traj, iters=iters, time_step= time_step, optim_name=optim_name, x0 =x_multi_0, method="NODE")
    print("NODE LE: ", LE_NODE)

    # Compute Jacobian Matrix and Lyapunov Exponent of rk4
    LE_rk4 = lyap_exps(True, longer_traj, iters=iters, time_step= time_step, optim_name=optim_name, x0 =x_multi_0, method="rk4")
    print("NODE rk4: ", LE_rk4)

    # Compute || LE_{NODE} - LE_{rk4} ||
    norm_difference = torch.linalg.norm(LE_NODE - LE_rk4)
    print("Norm Difference: ", norm_difference)


if __name__ == '__main__':
    # arguments (hyperparameters)
    # create_data (from your true model)
    # create_model (Neural ode) -- only this changes when you change from Lorenz to sth else
    
    # train the model, return node
    # calculate lyap_exps for true and node and compare
    main()