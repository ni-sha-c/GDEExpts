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
    X, Y, X_test, Y_test, g = sol.create_data(0, 25, torch.Tensor([0., 1.]), 10501, n_train=1000, n_test=3000, n_nodes=2, n_trans=0)

    ##### create dataloader #####
    train_iter, test_iter = sol.data_loader(X, Y, X_test, Y_test)
    print("created data!")

    ##### create model #####
    m = sol.create_NODE(device, n_nodes=2)
    print("created model!")

    ##### train #####

    num_epoch = 4000
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay =5e-4)

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
    loss_csv = np.asarray(loss_hist)
    np.savetxt("training_loss_sin.csv", loss_csv, delimiter=",")

    ##### Plot Phase Space #####
    Phase_space = plt.figure(figsize=(40,10))
    plt.style.use('_mpl-gallery')
    x_test = list(range(0, len(pred_test)))

    plt.scatter(x_test, pred_test[:, 0]) # num_test_data x 1 x num_node
    plt.scatter(x_test, X_test[:, 0])
    plt.legend(["Pred", "True"])
    plt.xticks()
    plt.yticks()
    plt.tight_layout()

    ##### Plot Time Space #####
    pred_train = np.array(pred_train)
    true_train = np.array(true_train)
    pred_test = np.array(pred_test)
    pred_train_last = pred_train[-1]
    true_train_last = true_train[-1]

    Time_space = plt.figure(figsize=(40,10))
    plt.style.use('_mpl-gallery')

    num_timestep = 1000
    x = list(range(0,num_timestep))
    x_loss = list(range(0,num_epoch))

    plt.subplot(2,2,1)
    plt.plot(x, pred_train_last[:num_timestep, 0 ], '--', linewidth=2)
    plt.plot(x, true_train_last[:num_timestep, 0])
    plt.plot(x, X[:num_timestep, 0])
    #plt.plot(x, x_train[:num_timestep, -1, substance_type], color='pink', alpha=0.7)

    plt.legend(['y_pred @ t + {}'.format(1), 'y_true @ t + {}'.format(1), 'x @ t + {}'.format(0)])
    plt.title('Chemical substance type A prediction at {} epoch, Train'.format(num_epoch))

    plt.subplot(2,2,2)
    plt.plot(pred_test[:num_timestep, 0], '--', linewidth=2)
    plt.plot(Y_test[:num_timestep, 0])
    plt.legend(['y_pred @ t + {}'.format(1), 'y_true @ t + {}'.format(1)])
    plt.title('Chemical substance type A prediction at {} epoch, Test'.format(num_epoch))

    ##### Plot Training Loss #####
    plt.subplot(2,2,3)
    plt.plot(x_loss, loss_hist)
    plt.title('Training Loss')
    
    plt.xticks()
    plt.yticks()
    plt.tight_layout()
    plt.show()

    return Phase_space, Time_space

    
phase, time= plot_attractor() 
plt.show()



