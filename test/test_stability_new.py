import argparse
from test_metrics import *
import datetime
import sys
import json
import ray
from ray import tune

sys.path.append('..')
from src.NODE_solve import *
# True Models
from examples.Brusselator import *
from examples.Lorenz import *
from examples.Lorenz_fixed import *
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
    # torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)


    # Set arguments (hyperparameters)
    DYNSYS_MAP = {'sin' : [sin, 1],
                  'tent_map' : [tent_map, 1],
                  'brusselator' : [brusselator, 2],
                  'lorenz_fixed' : [lorenz_fixed, 3],
                  'lorenz_periodic' : [lorenz_periodic, 3],
                  'lorenz' : [lorenz, 3]}

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=4000) # 10000
    parser.add_argument("--integration_time", type=int, default=110) #100
    parser.add_argument("--num_train", type=int, default=5000) #3000
    parser.add_argument("--num_test", type=int, default=5000)#3000
    parser.add_argument("--num_trans", type=int, default=100) #10000
    parser.add_argument("--iters", type=int, default=5*(10**4))
    parser.add_argument("--minibatch", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--loss_type", default="Jacobian", choices=["Jacobian", "MSE", "Auto_corr"])
    parser.add_argument("--reg_param", type=float, default=1e-5)
    parser.add_argument("--optim_name", default="AdamW", choices=["AdamW", "Adam", "RMSprop", "SGD"])
    parser.add_argument("--dyn_sys", default="lorenz", choices=DYNSYS_MAP.keys())

    args = parser.parse_args()
    dyn_sys_func, dim = define_dyn_sys(args.dyn_sys)
    dyn_sys_info = [dyn_sys_func, dim]
    if args.dyn_sys == "lorenz":
        rho = 28.0
    elif args.dyn_sys == "lorenz_periodic":
        rho = 350.
    else:
        rho = 0.8
    print("args: ", args)
    print("dyn_sys_func: ", dyn_sys_func)

    # Save args
    timestamp = datetime.datetime.now()
    # Convert the argparse.Namespace object to a dictionary
    args_dict = vars(args)

    # Assign Initial Point of Orbit
    x0 = torch.randn(dim)
    x_multi_0 = torch.randn(dim)
    print("initial point:", x_multi_0)

    # Initialize Model and Dataset Parameters
    criterion = torch.nn.MSELoss()
    real_time = args.iters * args.time_step
    print("real time: ", real_time)

    # Generate Training/Test/Multi-Step Prediction Data
    whole_traj = simulate(dyn_sys_func, 0, args.integration_time+1, x0, args.time_step) # last 100 points are for testing

    # After the transition phase, take 100 point
    # the first 100 point traj will be used to make S and S_diff
    # longer_traj is for the loss compuation

    S = []
    S_diff = []
    abs_loss_100 = []
    abs_loss_100_MSE = []
    # You take a random initial point and run a trajectory for many iterations (at least time 1). Take the point on the orbit after the runup(transition) time and the next 99 points on the same orbit as your training data.

    # longer_traj = simulate(dyn_sys_func, 0, real_time, x_multi_0, args.time_step)
    longer_traj = None
    trans = args.num_trans
    initial_points = []
    initial_points_MSE = []


    for i in range(10):
        print("------------ round ", i, " ------------")
        
        with open("stability_"+str(args.loss_type)+"_"+str(i)+'.txt', 'w') as f:
            json.dump(args_dict, f, indent=2)

        # Create S_i
        original_s = whole_traj[trans+i:trans+args.num_train+args.num_test+i+2, :]
        initial_points.append(original_s[0])
        initial_points_MSE.append(original_s[0])

        # Create S_i'
        x_m_diff = torch.randn(dim)
        y_m_diff = simulate(dyn_sys_func, 0, 0.011, x_m_diff, args.time_step)[-1] # (x_m', y_m') is y_m' = \varphi(x_m')

        revised_s = torch.clone(original_s)
        # revised_s[args.num_train-1, :] = x_m_diff
        revised_s[args.num_train-1, :] = x_m_diff
        revised_s[args.num_train, :] = y_m_diff

        print("new datapoint: ", x_m_diff, y_m_diff)
        print("org: ", original_s[args.num_train-2:args.num_train+1])
        print("rev: ", revised_s[args.num_train-2:args.num_train+1])

        # Data split: Create X, Y, X_test, Y_test
        original_dataset = create_data(original_s, n_train=args.num_train, n_test=args.num_test, n_nodes=dim, n_trans=0)
        revised_dataset = create_data(revised_s, n_train=args.num_train, n_test=args.num_test, n_nodes=dim, n_trans=0)

        X, Y, _, _ = original_dataset
        X_revised, Y_revised, X_test_revised, Y_test_revised = revised_dataset
        # Because of the way S' trajectory was created, have to make X_revised[-2] match with X[-2].
        print(X[-1, :].numpy())
        X_revised[-1, :] = torch.tensor(X[-1, :].clone().numpy())
        revised_dataset = X_revised, Y_revised, X_test_revised, Y_test_revised
        print("sanity check (x_m, y_m): ", X[-2:], Y[-2:])
        print("sanity check (x_m', y_m'): ",X_revised[-2:], Y_revised[-2:])
        print("shape", X.shape, X_revised.shape)

        S.append(original_s)
        S_diff.append(revised_s)

        # Create new model
        m = create_NODE(device, args.dyn_sys, n_nodes=dim, n_hidden=64,T=args.time_step).double()
        m_diff = create_NODE(device, args.dyn_sys, n_nodes=dim, n_hidden=64,T=args.time_step).double()
        m_MSE = create_NODE(device, args.dyn_sys, n_nodes=dim, n_hidden=64,T=args.time_step).double()
        m_MSE_diff = create_NODE(device, args.dyn_sys, n_nodes=dim, n_hidden=64,T=args.time_step).double()

        # Train the model, return node
        # if args.loss_type == "Jacobian":
        pred_train, true_train, pred_test, loss_hist, test_loss_hist, multi_step_error = jac_train(args.dyn_sys, m, device, original_dataset, longer_traj, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.time_step, real_time, args.num_trans, rho, args.reg_param, multi_step=False, minibatch=args.minibatch, batch_size=args.batch_size)

        pred_train_diff, true_train_diff, pred_test_diff, loss_hist_diff, test_loss_hist_diff, multi_step_error_diff = jac_train(args.dyn_sys, m_diff, device, revised_dataset, longer_traj, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.time_step, real_time, args.num_trans, rho, args.reg_param, multi_step = False,minibatch=args.minibatch, batch_size=args.batch_size)
            
        # else:
        pred_train_MSE, true_train_MSE, pred_test_MSE, loss_hist_MSE, test_loss_hist_MSE, multi_step_error_MSE = MSE_train(args.dyn_sys, m_MSE, device, original_dataset, longer_traj, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.time_step, real_time, args.num_trans, multi_step = False, minibatch=args.minibatch, batch_size=args.batch_size)

        pred_train_MSE_diff, true_train_MSE_diff, pred_test_MSE_diff, loss_hist_MSE_diff, test_loss_hist_MSE_diff, multi_step_error_MSE_diff = MSE_train(args.dyn_sys, m_MSE_diff, device, revised_dataset, longer_traj, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.time_step, real_time, args.num_trans, multi_step = False, minibatch=args.minibatch, batch_size=args.batch_size)


        # Maximum weights
        print("Saving Results...")

        lh = loss_hist[-1].tolist()
        tl = test_loss_hist[-1]
        lh_diff = loss_hist_diff[-1].tolist()
        tl_diff = test_loss_hist_diff[-1]

        lh_MSE = loss_hist_MSE[-1].tolist()
        tl_MSE = test_loss_hist_MSE[-1]
        lh_MSE_diff = loss_hist_MSE_diff[-1].tolist()
        tl_MSE_diff = test_loss_hist_MSE_diff[-1]

        abs_loss = torch.abs(torch.tensor(tl-tl_diff))
        abs_loss_100.append(abs_loss)

        abs_loss_MSE = torch.abs(torch.tensor(tl_MSE-tl_MSE_diff))
        abs_loss_100_MSE.append(abs_loss_MSE)

        with open('stability_'+str(args.loss_type)+"_"+str(i)+'.txt', 'a') as f:
            entry = {'train loss': lh, 
            'test loss': tl,
            "train loss S'": lh_diff, 
            "test loss S'": tl_diff, 
            "|loss S - loss S'|": abs_loss.detach().cpu().tolist(),
            'train loss MSE': lh_MSE, 
            'test loss MSE': tl_MSE,
            "train loss S' MSE": lh_MSE_diff, 
            "test loss S' MSE": tl_MSE_diff, 
            "|loss S - loss S'| in MSE": abs_loss_MSE.detach().cpu().tolist()}
            json.dump(entry, f)

    # Save abs loss
    loss_JAC = '../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"loss_100_"+"Jacobian"+".csv"
    loss_MSE = '../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"loss_100_"+"MSE"+".csv"

    if os.path.exists(loss_JAC):
        print("Data appended to existing file.")
        f_loss = open(loss_JAC,'a')
        np.savetxt(f_loss, np.asarray(abs_loss_100), delimiter=",")
    else:
        print("New file")
        np.savetxt(loss_JAC, np.asarray(abs_loss_100), delimiter=",")
    
    if os.path.exists(loss_MSE):
        print("Data appended to existing file.")
        f_MSE = open(loss_MSE,'a')
        np.savetxt(f_MSE, np.asarray(abs_loss_100_MSE), delimiter=",")
    else:
        print("New file")
        np.savetxt(loss_MSE, np.asarray(abs_loss_100_MSE), delimiter=",")

    
    JAC_filename = '../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"loss_100_initial_points"+"Jacobian"+".csv"

    MSE_filename = '../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"loss_100_initial_points"+"MSE"+".csv"

    if os.path.exists(JAC_filename):
        print("Data appended to existing file.")
        f = open(JAC_filename,'a')
        np.savetxt(f, np.asarray(initial_points), delimiter=",")
    else:
        print("New file")
        np.savetxt(JAC_filename, np.asarray(initial_points), delimiter=",")

    if os.path.exists(MSE_filename):
        print("Data appended to existing file.")
        f_MSE_init = open(MSE_filename,'a')
        np.savetxt(f_MSE_init, np.asarray(initial_points_MSE), delimiter=",")
    else:
        print("New file")
        np.savetxt(MSE_filename, np.asarray(initial_points_MSE), delimiter=",")

    print("Mean of 100 loss", np.mean(abs_loss_100))
    print("Mean of 100 loss", np.mean(abs_loss_100_MSE))