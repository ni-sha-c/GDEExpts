import argparse
from test_metrics import *
import sys
sys.path.append('..')
from src.NODE_solve import *
# True Models
from examples.Brusselator import *
from examples.Lorenz import *
from examples.Lorenz_periodic import *
from examples.Sin import *
from examples.Tent_map import *


if __name__ == '__main__':

    ''' Minimum Working Example of Neural ODE
            param:  dyn_system: function of dynamical system of interest
                    dim: dimension of state 
                    args: hyperparameters of model and dataset
            return: Lyapunov Exponents of true system and model ''' 

    # Set device
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)


    # Set arguments (hyperparameters)
    DYNSYS_MAP = {'sin' : [sin, 1],
                  'tent_map' : [tent_map, 1],
                  'brusselator' : [brusselator, 2],
                  'lorenz_periodic' : [lorenz_periodic, 3],
                  'lorenz' : [lorenz, 3]}

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=20000)
    parser.add_argument("--integration_time", type=int, default=180)
    parser.add_argument("--num_train", type=int, default=10000)
    parser.add_argument("--num_test", type=int, default=7500)
    parser.add_argument("--tran_state", type=int, default=100)
    parser.add_argument("--iters", type=int, default=5*(10**4))
    parser.add_argument("--minibatch", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--optim_name", default="AdamW", choices=["AdamW", "Adam", "RMSprop", "SGD"])
    parser.add_argument("--dyn_sys", default="lorenz", choices=DYNSYS_MAP.keys())

    args = parser.parse_args()
    dyn_sys_info = DYNSYS_MAP[args.dyn_sys]
    dyn_sys_func, dim = dyn_sys_info
    print("args: ", args)
    print("dyn_sys_func: ", dyn_sys_func)

    # Assign Initial Point of Orbit
    x0 = torch.randn(dim)
    x_multi_0 = torch.randn(dim)
    print("initial point:", x_multi_0)
    x1 = torch.tensor([0.1, 0.1, 0.1]).double()

    # Initialize Model and Dataset Parameters
    criterion = torch.nn.MSELoss()
    real_time = args.iters * args.time_step
    print("real time: ", real_time)

    # Generate Training/Test/Multi-Step Prediction Data
    traj = simulate(dyn_sys_func, 0, args.integration_time, x0, args.time_step)
    #longer_traj = simulate(dyn_sys_func, 0, real_time, x_multi_0, args.time_step)
    longer_traj = simulate(dyn_sys_func, 0, real_time, x1, args.time_step)
    multistep_traj = longer_traj[args.tran_state:]
    dataset = create_data(traj, n_train=args.num_train, n_test=args.num_test, n_nodes=dim, n_trans=args.tran_state)

    # Create model
    m = create_NODE(device, args.dyn_sys, n_nodes=dim, n_hidden=64,T=args.time_step).double()
     
    # Train the model, return node
    pred_train, true_train, pred_test, loss_hist, test_loss_hist = train(args.dyn_sys, m, device, dataset, multistep_traj, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.time_step, real_time, args.tran_state, minibatch=args.minibatch, batch_size=args.batch_size)

    print("train loss: ", loss_hist[-1])
    print("test loss: ", test_loss_hist[-1])


    # Save Trained Model
    path = "../test_result/expt_"+str(args.dyn_sys)+"/"+args.optim_name+"/"+str(args.time_step)+'/'+'model.pt'
    torch.save(m.state_dict(), path)

    # Save Training/Test Loss
    np.savetxt('../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"training_loss.csv", np.asarray(loss_hist), delimiter=",")
    np.savetxt('../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"test_loss.csv", np.asarray(test_loss_hist), delimiter=",")

    # Compute Jacobian Matrix and Lyapunov Exponent of Neural ODE
    LE_NODE, u = lyap_exps(args.dyn_sys, dyn_sys_info, longer_traj, iters=args.iters, time_step= args.time_step, optim_name=args.optim_name, method="NODE")
    print("NODE LE: ", LE_NODE)

    # Compute Jacobian Matrix and Lyapunov Exponent of rk4
    LE_rk4, u = lyap_exps(args.dyn_sys, dyn_sys_info, longer_traj, iters=args.iters, time_step= args.time_step, optim_name=args.optim_name, method="rk4")
    print("rk4 LE: ", LE_rk4)

    # Compute || LE_{NODE} - LE_{rk4} ||
    norm_difference = torch.linalg.norm(LE_NODE - LE_rk4)
    print("Norm Difference: ", norm_difference)