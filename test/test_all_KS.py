import argparse
from test_metrics import *
import datetime
import sys
import json
import ray
from ray import tune
from test_KS import *
import nolds

sys.path.append('..')
from src.NODE_solve import *
# True Models
from examples.Brusselator import *
from examples.Lorenz import *
from examples.Lorenz_fixed import *
from examples.Lorenz_periodic import *
from examples.Sin import *
from examples.Tent_map import *
from examples.KS import *
from examples.Henon import *


if __name__ == '__main__':

    # python test_all_KS.py --dyn_sys=KS --loss_type=MSE --time_step=0.1 --iters=0

    # Set device
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)


    # Set arguments (hyperparameters)
    DYNSYS_MAP = {'sin' : [sin, 1],
                  'tent_map' : [tent_map, 1],
                  'KS': [run_KS, 128],
                  'henon': [henon, 2],
                  'brusselator' : [brusselator, 2],
                  'lorenz_fixed' : [lorenz_fixed, 3],
                  'lorenz_periodic' : [lorenz_periodic, 3],
                  'lorenz' : [lorenz, 3]}

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_epoch", type=int, default=10000) # 10000
    parser.add_argument("--integration_time", type=int, default=0) #100
    parser.add_argument("--num_train", type=int, default=1000) #3000
    parser.add_argument("--num_test", type=int, default=800)#3000
    parser.add_argument("--num_trans", type=int, default=0) #10000
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--minibatch", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--loss_type", default="MSE", choices=["Jacobian", "MSE", "Auto_corr"])
    parser.add_argument("--reg_param", type=float, default=1e-6)
    parser.add_argument("--optim_name", default="AdamW", choices=["AdamW", "Adam", "RMSprop", "SGD"])
    parser.add_argument("--dyn_sys", default="KS", choices=DYNSYS_MAP.keys())

    args = parser.parse_args()
    dyn_sys_func, dim = define_dyn_sys(args.dyn_sys)
    dyn_sys_info = [dyn_sys_func, args.dyn_sys, dim]
    
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
    with open(str(timestamp)+'.txt', 'w') as f:
        json.dump(args_dict, f, indent=2)

    # Assign Initial Point of Orbit
    # L = 256 # n = [128, 256, 512, 700]
    L = 128
    n = L-1 # num of internal node
    T = 10 #1000 #100
    c = 0.4

    dx = L/(n+1)
    dt = args.time_step
    x = torch.arange(0, L+dx, dx) # [0, 0+dx, ... 128] shape: L + 1
    u0 = 2.71828**(-(x-64)**2/512).to(device).double()
    # torch.exp(-(x-64)**2/512)
    # u_multi_0 = -0.5 + torch.rand(n+2)

    # Initialize Model and Dataset Parameters
    criterion = torch.nn.MSELoss()
    real_time = args.iters * args.time_step
    print("real time: ", real_time)

    # boundary condition
    u0[0], u0[-1] = 0, 0 
    u0 = u0.requires_grad_(True)

    # Generate Training/Test/Multi-Step Prediction Data
    u_list = run_KS(u0, c, dx, dt, T, False, device)

    plot_KS(u_list, dx, L+1, c, T, dt, True, False)

    u_list = u_list[:, :-1] # remove the last boundary node and keep the first boundary node as it is initial condition

    print('u0', u_list[:, 0])
    print("u", u_list.shape)

    # Data split
    x_grid_idx = 1
    # dataset = create_data(u_list[:, x_grid_idx], n_train=args.num_train, n_test=args.num_test, n_nodes=dim, n_trans=args.num_trans)
    dataset = create_data(u_list, n_train=args.num_train, n_test=args.num_test, n_nodes=dim, n_trans=args.num_trans)

    X, Y, x_test, y_test = dataset
    print(X.shape, Y.shape, x_test.shape, y_test.shape)

    # Create model
    m = create_NODE(device, args.dyn_sys, n_nodes=dim, n_hidden=64,T=args.time_step).double()
    longer_traj = None

    # Train the model, return node
    if args.loss_type == "Jacobian":
        pred_train, true_train, pred_test, loss_hist, test_loss_hist, multi_step_error = jac_train(dyn_sys_info, m, device, dataset, longer_traj, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.time_step, real_time, args.num_trans, rho, args.reg_param, multi_step=True, minibatch=args.minibatch, batch_size=args.batch_size)

    elif args.loss_type == "Auto_corr":
        pred_train, true_train, pred_test, loss_hist, test_loss_hist, multi_step_error = ac_train(args.dyn_sys, m, device, dataset, longer_traj, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.time_step, real_time, args.num_trans, rho, minibatch=args.minibatch, batch_size=args.batch_size)
        
    else:
        pred_train, true_train, pred_test, loss_hist, test_loss_hist, multi_step_error = MSE_train(dyn_sys_info, m, device, dataset, longer_traj, args.optim_name, criterion, args.num_epoch, args.lr, args.weight_decay, args.time_step, real_time, args.num_trans, multi_step=False, minibatch=args.minibatch, batch_size=args.batch_size)

    


    # Maximum weights
    print("Saving Results...")
    max_weight = []
    for param_tensor in m.state_dict():
        if "weight" in param_tensor:
            weights = m.state_dict()[param_tensor].squeeze()
            max_weight.append(torch.max(weights).cpu().tolist())

    # Maximum solution
    pred_train = torch.tensor(np.array(pred_test))
    true_train = torch.tensor(np.array(y_test))
    print(pred_train.shape, true_train.shape)

    # plot_KS(pred_train, dx, L, c, T/4, dt, False, True)
    # plot_KS(true_train[-1], dx, L, c, T/4, dt, True, False)
    
    max_solution = [torch.max(pred_train[:, 0]).cpu().tolist(), torch.max(pred_train[:, 1]).cpu().tolist(), torch.max(pred_train[:, 2]).cpu().tolist()]

    # Dump Results
    if torch.isnan(multi_step_error):
        multi_step_error = torch.tensor(0.)

    lh = loss_hist[-1].tolist()
    tl = test_loss_hist[-1]
    ms = max_solution
    mw = max_weight
    mse = multi_step_error.cpu().tolist()

    with open(str(timestamp)+'.txt', 'a') as f:
        entry = {'train loss': lh, 'test loss': tl, 'max of solution': ms, 'max of weight': mw, 'multi step prediction error': mse}
        json.dump(entry, f)


    # Save Trained Model
    model_path = "../test_result/expt_"+str(args.dyn_sys)+"/"+args.optim_name+"/"+str(args.time_step)+'/'+'model.pt'
    torch.save(m.state_dict(), model_path)
    print("Saved new model!")


    # Save whole trajectory
    # np.savetxt('../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"whole_traj.csv", np.asarray(whole_traj.detach().cpu()), delimiter=",")

    # Save Training/Test Loss
    loss_hist = torch.stack(loss_hist)
    np.savetxt('../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"training_loss.csv", np.asarray(loss_hist.detach().cpu()), delimiter=",")
    np.savetxt('../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"test_loss.csv", np.asarray(test_loss_hist), delimiter=",")

    # save trajectory
    # np.savetxt('../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"pred.csv", pred_train.reshape(pred_train.shape[0], -1).detach().cpu().numpy(), delimiter=",")
    # np.savetxt('../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"true.csv", true_train.reshape(true_train.shape[0], -1).detach().cpu().numpy(), delimiter=",")

    '''LE_NODE = lyap_exps_ks(args.dyn_sys, dyn_sys_info, u_list, T*int(1/dt), u_list, dx, L, c, T, dt, time_step= args.time_step, optim_name=args.optim_name, method="NODE", path=model_path)'''
    # LE_NODE = nolds.lyap_e(pred_train[-1, :, -1], emb_dim=19, matrix_dim=10, min_nb=None, min_tsep=0, tau=1, debug_plot=False, debug_data=False, plot_file=None)
    # print("NODE LE: ", LE_NODE)

    # Compute Jacobian Matrix and Lyapunov Exponent of rk4
    # u_list = np.array(u_list.detach().cpu())

    LE_rk4 = lyap_exps_ks(args.dyn_sys, dyn_sys_info, u_list, T*int(1/dt), u_list, dx, L, c, T, dt, time_step= args.time_step, optim_name=args.optim_name, method="rk4", path=model_path)
    print("rk4 LE: ", LE_rk4)
    LE_rk4 = nolds.lyap_e(true_train[-1, :], emb_dim=19, matrix_dim=10, min_nb=None, min_tsep=0, tau=1, debug_plot=False, debug_data=False, plot_file=None)
    print("rk4 LE: ", LE_rk4)

    # Compute || LE_{NODE} - LE_{rk4} ||
    norm_difference = torch.linalg.norm(torch.tensor(LE_NODE) - torch.tensor(LE_rk4))
    print("Norm Difference: ", norm_difference)

    # with open(str(timestamp)+'.txt', 'a') as f:
    entry = {'Nerual ODE LE': LE_NODE.detach().cpu().tolist(), 'rk4 LE': LE_rk4.detach().cpu().tolist(), 'norm difference': norm_difference.detach().cpu().tolist()}
    json.dump(entry, f)