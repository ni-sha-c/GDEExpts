import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append('..')

from src import NODE_solve_Brusselator as sol 

def plot_attractor():
    ''' func: plotting the attractor '''
    # train loss:  5.7104885118373735e-08
    # test loss:  5.995462423718316e-08

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)

    ##### create data #####
    X, Y, X_test, Y_test = sol.create_data(0, 800, torch.Tensor([1,3]), 40001, n_train=2000, n_test=1000, n_nodes=2, n_trans=37000)
    
    ##### create data for extrapolation #####
    true_traj = sol.simulate(800, 1600, torch.Tensor([1,3]), 40001)
    true_traj = true_traj[37000:]
    print("testing initial point: ", true_traj[0])
    print("created data!")

    ##### create model #####
    m = sol.create_NODE(device, n_nodes=2)
    print("created model!")

    ##### train #####
    num_epoch = 4000
    criterion = torch.nn.MSELoss()
    lr=5e-4
    optimizer = torch.optim.RMSprop(m.parameters(), lr=lr, weight_decay =5e-4) # 1e-4

    pred_train, true_train, pred_test, loss_hist, test_loss_hist = sol.train(m,
                                                                             device,
                                                                             X,
                                                                             Y,
                                                                             X_test,
                                                                             Y_test, 
                                                                             true_traj,
                                                                             optimizer,
                                                                             criterion,
                                                                             epochs=num_epoch)
    print("train loss: ", loss_hist[-1])
    print("test loss: ", test_loss_hist[-1])

    ##### Save Training Loss #####
    optim_name = 'RMSprop'
    loss_csv = np.asarray(loss_hist)
    np.savetxt('expt_brusselator/'+ optim_name + '/' + "training_loss.csv", loss_csv, delimiter=",")

    ##### Plot Phase Space #####
    plt.figure(figsize=(40,10))
    plt.scatter(pred_test[:, 0], pred_test[:, 1]) # num_test_data x 1 x num_node
    plt.scatter(Y_test[:, 0], Y_test[:, 1])
    plt.legend(['Pred', 'True'])
    plt.xticks()
    plt.yticks()
    plt.savefig('expt_brusselator/' + optim_name + '/' + 'Phase Space with ' + 'lr=' + str(lr), format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    plt.show()
    plt.close()

    ##### Plot Time Space #####
    pred_train = np.array(pred_train)
    true_train = np.array(true_train)
    pred_test = np.array(pred_test)
    pred_train_last = pred_train[-1]
    true_train_last = true_train[-1]

    plt.figure(figsize=(40,10))

    num_timestep = 2000
    substance_type = 0
    x = list(range(0,num_timestep))
    x_loss = list(range(0,num_epoch))

    plt.subplot(2,2,1)
    plt.plot(x, pred_train_last[:num_timestep, substance_type], '--', linewidth=2)
    plt.plot(x, true_train_last[:num_timestep, substance_type], alpha=0.7)
    plt.plot(x, X[:num_timestep, substance_type], c='gray', alpha=0.5)
    plt.legend(['y_pred @ t + {}'.format(1), 'y_true @ t + {}'.format(1), 'x @ t + {}'.format(0)])
    plt.title('Chemical substance type A prediction at {} epoch, Train'.format(num_epoch))

    plt.subplot(2,2,2)
    plt.plot(pred_test[:num_timestep, substance_type], '--', linewidth=2)
    plt.plot(Y_test[:num_timestep, substance_type])
    plt.plot(X_test[:num_timestep, substance_type])
    plt.legend(['y_pred @ t + {}'.format(1), 'y_true @ t + {}'.format(1), 'x @ t + {}'.format(0)])
    plt.title('Chemical substance type A prediction at {} epoch, Test'.format(num_epoch))

    ##### Plot Training Loss #####
    plt.subplot(2,2,3)
    plt.plot(x_loss, loss_hist)
    plt.title('Training Loss')
    plt.xticks()
    plt.yticks()
    plt.savefig('expt_brusselator/' + optim_name + '/' + 'Time Space, Training Loss, Test Loss with ' + 'lr=' + str(lr), format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    plt.show()
    plt.close("all")

    return 


##### run experiment #####    
plot_attractor() 




