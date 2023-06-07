import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append('..')

from src import solve_ode_with_nn as sol 

def plot_attractor():
    ''' func: plotting the attractor '''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ##### create data #####
    X, Y, X_test, Y_test, g = sol.create_data(0, 800, torch.Tensor([1,3]), 4001, n_train=200, n_test=50, n_nodes=2, n_trans=3500)
    ##### modify g #####
    g = sol.modify_graph(g, device)
    ##### create dataloader #####
    train_iter, test_iter = sol.data_loader(X, Y, X_test, Y_test)
    print("created data!")

    ##### create model #####
    m = sol.create_NODE(device)
    print("created model!")

    ##### train #####
    # call loss, test_loss, model_final = train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=7e-2, weight_decay =5e-4)

    pred_train, true_train, x_train, pred_test, true_test, loss_hist, test_loss_hist = sol.train(m,
                                                                                             device,
                                                                                             train_iter,
                                                                                             test_iter,
                                                                                             optimizer,
                                                                                             criterion,
                                                                                             epochs=3500)
    print("train loss: ", loss_hist[-1])
    print("test loss: ", test_loss_hist[-1])

    ##### plot attractor #####
    fig = plt.figure(figsize=(40,10))
    plt.style.use('_mpl-gallery')
    plt.scatter(pred_test[:, -1, 0], pred_test[:, -1, 1]) # num_test_data x 1 x num_node
    plt.xticks()
    plt.yticks()
    plt.tight_layout()
    return fig


    #x0 = random.rand(2)
    #for t in range(t_plot):
        # call model_final(x)
        # call the ode solver
        # plot attractor
    
fig = plot_attractor() 
plt.show()



