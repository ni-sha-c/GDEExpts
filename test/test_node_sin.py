import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append('..')

from src import NODE_solve_Sin as sol 

def plot_attractor():
    ''' func: plotting the attractor '''
    # train loss:  3.065560933961977e-13
    # test loss:  2.933217361852054e-08

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ##### create data #####
    X, Y, X_test, Y_test = sol.create_data(0, 25, torch.Tensor([0., 1.]), 10001, n_train=6000, n_test=3000, n_nodes=2, n_trans=0)
    print("created data!")

    ##### create model #####
    m = sol.create_NODE(device, n_nodes=2)
    print("created model!")

    ##### train #####
    num_epoch = 4000
    lr = 5e-4
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(m.parameters(), lr=lr, weight_decay =5e-4, momentum=0)

    pred_train, true_train, pred_test, loss_hist, test_loss_hist = sol.train(m,
                                                                             device,
                                                                             X,
                                                                             Y,
                                                                             X_test,
                                                                             Y_test,
                                                                             optimizer,
                                                                             criterion,
                                                                             epochs=num_epoch)
    
    print("train loss: ", loss_hist[-1])
    print("test loss: ", test_loss_hist[-1])

    ##### Save Training Loss #####
    optim_name = 'Gradient Descent'
    loss_csv = np.asarray(loss_hist)
    np.savetxt('expt_sin_second_order/'+ optim_name + '/' + "training_loss_sin.csv", loss_csv, delimiter=",")

    ##### Plot Phase Space #####
    plt.figure(figsize=(40,10))
    x_test = list(range(0, len(pred_test)))

    plt.scatter(x_test, pred_test[:, 0]) # num_test_data x 1 x num_node
    plt.scatter(x_test, Y_test[:, 0])
    plt.legend(["Pred", "True"])
    plt.xticks()
    plt.yticks()
    plt.tight_layout()
    plt.savefig('expt_sin_second_order/' + optim_name + '/' + 'Phase Space with ' + 'lr=' + str(lr), format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    plt.show()
    plt.close()


    ##### Plot Time Space #####
    pred_train = np.array(pred_train)
    true_train = np.array(true_train)
    pred_test = np.array(pred_test)
    pred_train_last = pred_train[-1]
    true_train_last = true_train[-1]

    plt.figure(figsize=(35,10))
    num_timestep = 3000
    x = list(range(0,num_timestep))
    x_loss = list(range(0,num_epoch))

    plt.subplot(2,2,1)
    plt.plot(x, pred_train_last[:num_timestep, 0], '--', linewidth=2)
    plt.plot(x, true_train_last[:num_timestep, 0], linewidth=1.5, alpha=0.7)
    plt.plot(x, X[:num_timestep, 0], c='gray', alpha=0.5, linewidth=1.5)
    plt.legend(['y_pred @ t + {}'.format(1), 'y_true @ t + {}'.format(1), 'x @ t + {}'.format(0)])
    plt.title('Sin prediction at {} epoch, Train'.format(num_epoch))

    plt.subplot(2,2,2)
    plt.plot(pred_test[:num_timestep, 0], '--', linewidth=2)
    plt.plot(Y_test[:num_timestep, 0])
    plt.legend(['y_pred @ t + {}'.format(1), 'y_true @ t + {}'.format(1)])
    plt.title('Sin prediction at {} epoch, Test'.format(num_epoch))

    ##### Plot Training Loss #####
    plt.subplot(2,2,3)
    plt.plot(x_loss, loss_hist)
    plt.title('Training Loss')
    plt.xticks()
    plt.yticks()
    plt.savefig('expt_sin_second_order/' + optim_name + '/' + 'Time Space, Training Loss, Test Loss with ' + 'lr=' + str(lr), format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    plt.show()
    plt.close("all")

    return

plot_attractor()




