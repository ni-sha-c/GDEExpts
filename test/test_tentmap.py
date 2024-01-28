import torch
from matplotlib.pyplot import *
import sys

sys.path.append('..')
from src.NODE_solve import *
from examples.Tent_map import *



def jac_train_tentmap(dyn_sys_info, model, device, dataset, true_t, optim_name, criterion, epochs, lr, weight_decay, time_step, real_time, tran_state, rho, reg_param, s, multi_step = False,minibatch=False, batch_size=0):

    # Initialize
    pred_train, true_train, loss_hist, test_loss_hist = ([] for i in range(4))
    optimizer = define_optimizer(optim_name, model, lr, weight_decay)
    X, Y, X_test, Y_test = dataset
    X, Y, X_test, Y_test = X.to(device), Y.to(device), X_test.to(device), Y_test.to(device)
    num_train = X.shape[0]

    # dyn_sys_info = [dyn_sys_func, dim]
    dyn_sys, dyn_sys_name, dim = dyn_sys_info
    
    # Compute True Jacobian
    t_eval_point = torch.linspace(0, time_step, 2).to(device).double()

    True_J = torch.ones(num_train, dim, dim).to(device)
    if dyn_sys_name == "lorenz":
        for i in range(num_train):
            True_J[i] = Jacobian_Matrix(X[i, :], sigma=10.0, r=rho, b=8/3)
    elif dyn_sys_name == "coupled_brusselator":
        print("Computing Jacobian of Brusselator!!")
        for i in range(num_train):
            True_J[i] = Jacobian_Brusselator(dyn_sys, X[i, :])
    elif dyn_sys_name == "henon":
        print("henon")
        for i in range(num_train):
            True_J[i] = Jacobian_Henon(X[i, :])
    elif dyn_sys_name == "baker":
        print("baker")
        for i in range(num_train):
            True_J[i] = F.jacobian(baker, X[i, :])

    elif dyn_sys_name == "tent_map":
        print("tent_map")
        for i in range(num_train):
            # True_J[i] = F.jacobian(tent_map, X[i, :].squeeze())
            c = X[i,:]
            t = tent_map(c, s)
            r = torch.autograd.grad(t, c)
            True_J[i] = r[0] #, create_graph=True
            print(True_J[i])

    print(True_J.shape)
    print("Finished Computing True Jacobian")

    for i in range(epochs): # looping over epochs
        model.train()
        model.double()

        if minibatch == True:
            return None

        elif minibatch == False:

            if (dyn_sys_name == "henon") or (dyn_sys_name == "baker") or (dyn_sys_name == "tent_map"):
                y_pred = model(X).to(device)
            else: 
                y_pred = solve_odefunc(model, t_eval_point, X).to(device)

            optimizer.zero_grad()
            # MSE Output Loss
            MSE_loss = criterion(y_pred, Y)

            # Jacobian Diff Loss
            if (dyn_sys_name == "henon") or (dyn_sys_name == "baker") or (dyn_sys_name == "tent_map"):
                jacrev = torch.func.jacrev(model)
                compute_batch_jac = torch.vmap(jacrev)
                cur_model_J = compute_batch_jac(X).to(device)
            else:
                jacrev = torch.func.jacrev(model, argnums=1)
                compute_batch_jac = torch.vmap(jacrev, in_dims=(None, 0))
                cur_model_J = compute_batch_jac(t_eval_point, X).to(device)
            train_loss = jacobian_loss(True_J, cur_model_J, MSE_loss, reg_param)
            train_loss.backward()
            optimizer.step()

            # leave it for debug purpose for now, and remove it
            pred_train.append(y_pred.detach().cpu().numpy())
            true_train.append(Y.detach().cpu().numpy())
        
        loss_hist.append(train_loss)
        if i % 1000 == 0:
            print(i, MSE_loss.item(), train_loss.item())

        ##### test one_step #####
        pred_test, test_loss = evaluate(dyn_sys_name, model, time_step, X_test, Y_test, device, criterion, i, optim_name)
        test_loss_hist.append(test_loss)

        ##### test multi_step #####
        if (i+1) == epochs and (multi_step == True):
            error = test_multistep(dyn_sys, dyn_sys_name, model, epochs, true_t, device, i, optim_name, lr, time_step, real_time, tran_state)
        else:
            error = torch.tensor(0.)

    return pred_train, true_train, pred_test, loss_hist, test_loss_hist, error




# savefig
fig, ax = subplots(figsize=(24,13))
pdf_path = '../plot/plucked_map_JAC'+'.jpg'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
T = 50000
T_min = 500
colors = cm.viridis(np.linspace(0, 1, 6))

# create True Plot
s_list = [0.1, 0.3, 0.5, 0.7, 0.9]
print(s_list)
x0 = 0.63
# x = torch.linspace(0.01, 1.99, T)
# x_min = torch.linspace(0.01, 1.99, T_min)

for idx, s in enumerate(s_list):

    # whole_traj = torch.zeros(T)
    # # x = x0
    # for i in range(T):
    #     next_x = tent_map(x[i], s)
    #     whole_traj[i] = next_x
    #     # x = next_x

    whole_traj = torch.zeros(T, 1)
    
    for i in range(T):
        next_x = tent_map(x0, s)
        whole_traj[i] = next_x
        x0 = next_x

    # Train model
    dataset = create_data(whole_traj, n_train=10000, n_test=8000, n_nodes=1, n_trans=0)
    x_t, y_t, x_test, y_test = dataset
    dataset = [x_t.requires_grad_(True), y_t.requires_grad_(True), x_test.requires_grad_(True), y_test.requires_grad_(True)]
    # dataset = [torch.tensor(x[:50000]).double().reshape(-1,1), torch.tensor(whole_traj[:50000]).double().reshape(-1,1), torch.tensor(x[50000:]).double().reshape(-1,1), torch.tensor(whole_traj[50000:]).double().reshape(-1,1)]
    m = create_NODE(device, "tent_map", n_nodes=1, n_hidden=64,T=1).double()
    dyn_sys_info = [tent_map, "tent_map", 1]
    longer_traj = []
    criterion = torch.nn.MSELoss()
    real_time = 500
    # MSE
    # pred_train, true_train, pred_test, loss_hist, test_loss_hist, multi_step_error = MSE_train(dyn_sys_info, m, device, dataset, longer_traj, "AdamW", criterion, 15000, 5e-4, 5e-4, 1, real_time, 0, multi_step=False, minibatch=False, batch_size=0)

    #JAc
    pred_train, true_train, pred_test, loss_hist, test_loss_hist, multi_step_error = jac_train_tentmap(dyn_sys_info, m, device, dataset, longer_traj, "AdamW", criterion, 20000, 5e-4, 5e-4, 1, real_time, 0, 28.0, 1e-6, s, multi_step=False, minibatch=False, batch_size=0)


    # Plot
    # traj = torch.zeros(T_min)

    # for j in range(T_min):

    #     input = torch.tensor(x_min[j]).reshape(1,-1).double().to(device)
    #     next_x = m(input)
    #     traj[j] = next_x


    ax.scatter(pred_test[0:-1].detach().cpu().numpy(), pred_test[1:].detach().cpu().numpy(), color=colors[idx], linewidth=6, alpha=0.8, label='s = ' + str(s))
    # print(whole_traj[-10:], traj[-10:])

# ax.grid(True)
ax.set_xlabel(r"$x_n$", fontsize=44)
ax.set_ylabel(r"$x_{n+1}$", fontsize=44)
ax.tick_params(labelsize=40)
ax.legend(loc='best', fontsize=40)
tight_layout()
            
fig.savefig(pdf_path, format='jpg', dpi=400)


