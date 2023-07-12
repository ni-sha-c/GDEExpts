import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append('..')

from src import NODE_solve_Lorenz as sol 

def plot_attractor():
    ''' func: plotting the attractor '''

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)

    ##### create data #####
    X, Y, X_test, Y_test = sol.create_data(0, 40, torch.Tensor([ -8., 7., 27.]), 80001, n_train=15000, n_test=5000, n_nodes=3, n_trans=60000)
    print("created data!")

    ##### create model #####
    m = sol.create_NODE(device, n_nodes=3)
    print("created model!")

    ##### train #####
    num_epoch = 10000
    criterion = torch.nn.MSELoss()
    lr=5e-4
    optimizer = torch.optim.SGD(m.parameters(), lr=lr, weight_decay =5e-4, momentum=0) # 1e-4

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
    np.savetxt('expt_lorenz/'+ optim_name + '/' + "training_loss.csv", loss_csv, delimiter=",")

    ##### Plot Phase Space #####
    plt.figure(figsize=(20,15))
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.plot3D(pred_test[:, 0], pred_test[:, 1], pred_test[:, 2], 'gray', linewidth=5)
        
    z = Y_test[:, 2]
    ax.scatter3D(Y_test[:, 0], Y_test[:, 1], z, c=z, cmap='hsv', alpha=0.3, linewidth=0)
    ax.set_title('Phase Space')
    plt.savefig('expt_lorenz/' + optim_name + '/' + 'Phase Space with ' + 'lr=' + str(lr), format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
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
    plt.plot(x, X[:num_timestep, substance_type], c='gray', alpha=0.7)
    plt.legend(['y_pred @ t + {}'.format(1), 'y_true @ t + {}'.format(1), 'x @ t + {}'.format(0)])
    plt.title('Substance type A prediction at {} epoch, Train'.format(num_epoch))

    plt.subplot(2,2,2)
    plt.plot(pred_test[:num_timestep, substance_type], '--', linewidth=2)
    plt.plot(Y_test[:num_timestep, substance_type])
    plt.plot(X_test[:num_timestep, substance_type])
    plt.legend(['y_pred @ t + {}'.format(1), 'y_true @ t + {}'.format(1), 'x @ t + {}'.format(0)])
    plt.title('Substance type A prediction at {} epoch, Test'.format(num_epoch))

    ##### Plot Training Loss #####
    plt.subplot(2,2,3)
    plt.plot(x_loss, loss_hist)
    plt.title('Training Loss')
    plt.xticks()
    plt.yticks()
    plt.savefig('expt_lorenz/' + optim_name + '/' + 'Time Space, Training Loss, Test Loss with ' + 'lr=' + str(lr), format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    plt.show()
    plt.close("all")

    return 


##### run experiment #####    
plot_attractor() 




