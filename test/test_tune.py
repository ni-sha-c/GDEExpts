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
    torch.manual_seed(42)
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
    parser.add_argument("--num_epoch", type=int, default=5000) # 10000
    parser.add_argument("--integration_time", type=int, default=200) #100
    parser.add_argument("--num_train", type=int, default=10000) #3000
    parser.add_argument("--num_test", type=int, default=8000) #3000
    parser.add_argument("--num_trans", type=int, default=0) #10000
    parser.add_argument("--iters", type=int, default=5*(10**4))
    parser.add_argument("--minibatch", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--loss_type", default="Jacobian", choices=["Jacobian", "MSE", "Auto_corr"])
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
    with open(str(timestamp)+'.txt', 'w') as f:
        json.dump(args_dict, f, indent=2)

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
    training_traj = whole_traj[:args.integration_time*int(1/args.time_step), :]
    longer_traj = simulate(dyn_sys_func, 0, real_time, x_multi_0, args.time_step)

    # Create model
    m = create_NODE(device, args.dyn_sys, n_nodes=dim, n_hidden=64,T=args.time_step).double()


    def grid_search_jac_train(config):
        assert torch.cuda.is_available()
        print(ray.get_gpu_ids())
        print("cuda available?:", torch.cuda.is_available())

        dataset = create_data(training_traj, n_train=args.num_train, n_test=args.num_test, n_nodes=dim, n_trans=config["trans_phase"])
        # reg 0.01, 100, [-9.3787, -9.9255, 27.5508]
        # reg 0.01, 200, [-7.3727, -7.7015, 25.1829]

        pred_train, true_train, pred_test, loss_hist, test_loss_hist, multi_step_error = hyperparam_gridsearch_MSE(
            args.dyn_sys, m, device, dataset, longer_traj, args.optim_name, criterion, args.num_epoch,
            args.lr, args.weight_decay, args.time_step, real_time, config["trans_phase"], rho,
            config["reg_param"], minibatch=args.minibatch, batch_size=args.batch_size
        )

        return test_loss_hist[-1]  #  metric to optimize

    # Train the model, return node

    # GridSearch for reg_param
    # Start Ray Head Node: $ray start --head
    
    ray.init(address='auto', 
    runtime_env={"working_dir": ".", 
    "excludes": ["/animation", "/archive_test", "/jacobian_loss", "/.git"],
    "py_modules": ["/storage/home/hcoda1/6/jpark3141/p-nisha3-0/GDEExpts/src", "/storage/home/hcoda1/6/jpark3141/p-nisha3-0/GDEExpts/examples", "/storage/home/hcoda1/6/jpark3141/p-nisha3-0/GDEExpts/test_result"]}, 
    log_to_driver=True, 
    logging_level=logging.DEBUG)

    analysis = tune.run(
    grid_search_jac_train,
    resources_per_trial={"cpu":32, "gpu": 2},
    config={"reg_param": tune.grid_search([1e-2, 1e-3, 1e-4, 1e-5, 1e-6]),
            "trans_phase": tune.grid_search([0, 100, 200, 300, 400, 500])})

    best_trial = analysis.get_best_trial("loss", "min", "last")
    # print(f"Best trial config: {best_trial}")

    df = analysis.dataframe()
    print(df)
    # Plot by epoch
    ax = None  # This plots everything on the same plot
    for d in df.values():
        ax = d.mean_accuracy.plot(ax=ax, legend=True)


    # Maximum weights
    print("Saving Results...")
    max_weight = []
    for param_tensor in m.state_dict():
        if "weight" in param_tensor:
            weights = m.state_dict()[param_tensor].squeeze()
            max_weight.append(torch.max(weights).cpu().tolist())

    # Maximum solution
    pred_train = torch.tensor(np.array(pred_train))
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
    np.savetxt('../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"whole_traj.csv", np.asarray(whole_traj.detach().cpu()), delimiter=",")

    # Save Training/Test Loss
    loss_hist = torch.stack(loss_hist)
    np.savetxt('../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"training_loss.csv", np.asarray(loss_hist.detach().cpu()), delimiter=",")
    np.savetxt('../test_result/expt_'+str(args.dyn_sys)+'/'+ args.optim_name + '/' + str(args.time_step) + '/' +"test_loss.csv", np.asarray(test_loss_hist), delimiter=",")

    # Compute Jacobian Matrix and Lyapunov Exponent of Neural ODE
    LE_NODE = lyap_exps(args.dyn_sys, dyn_sys_info, longer_traj, iters=args.iters, time_step= args.time_step, optim_name=args.optim_name, method="NODE", path=model_path)
    print("NODE LE: ", LE_NODE)

    # Compute Jacobian Matrix and Lyapunov Exponent of rk4
    LE_rk4 = lyap_exps(args.dyn_sys, dyn_sys_info, longer_traj, iters=args.iters, time_step= args.time_step, optim_name=args.optim_name, method="rk4", path=model_path)
    print("rk4 LE: ", LE_rk4)

    # Compute || LE_{NODE} - LE_{rk4} ||
    norm_difference = torch.linalg.norm(LE_NODE - LE_rk4)
    print("Norm Difference: ", norm_difference)

    with open(str(timestamp)+'.txt', 'a') as f:
        entry = {'Nerual ODE LE': LE_NODE.detach().cpu().tolist(), 'rk4 LE': LE_rk4.detach().cpu().tolist(), 'norm difference': norm_difference.detach().cpu().tolist()}
        json.dump(entry, f)