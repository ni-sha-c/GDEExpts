import matplotlib.pyplot as plt
import numpy as np

def plot_traj_lorenz(X, optim_name, time):
    '''plot trajectory of lorenz training data'''

    plt.plot(X[:, 0], color="C1")
    plt.plot(X[:, 1], color="C2")
    plt.plot(X[:, 2], color="C3")

    plt.title('Trajectory of Training Data')
    plt.xticks()
    plt.yticks()
    plt.legend(["X", "Y", "Z"])
    plt.savefig('expt_lorenz/' + optim_name + '/' + str(time) + '/' + 'train_data_traj', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)

    return



def plot_phase_space_lorenz(pred_test, Y_test, optim_name, lr, time):
    '''plot phase space of lorenz'''

    plt.figure(figsize=(20,15))
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.plot3D(pred_test[:, 0], pred_test[:, 1], pred_test[:, 2], 'gray', linewidth=5)
        
    z = Y_test[:, 2]
    ax.scatter3D(Y_test[:, 0], Y_test[:, 1], z, c=z, cmap='hsv', alpha=0.3, linewidth=0)
    ax.set_title('Phase Space')
    plt.savefig('expt_lorenz/' + optim_name + '/' + str(time) + '/' + 'Phase Space with ' + 'lr=' + str(lr), format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    plt.show()
    plt.close("all")

    return


def plot_time_space_lorenz(X, X_test, Y_test, pred_train, true_train, pred_test, loss_hist, optim_name, lr, num_epoch, time_step):
    '''plot time_space for training/test data and training loss for lorenz system'''

    pred_train = np.array(pred_train)
    true_train = np.array(true_train)
    pred_test = np.array(pred_test)
    pred_train_last = pred_train[-1]
    true_train_last = true_train[-1]

    plt.figure(figsize=(40,10))

    num_timestep = 1500
    substance_type = 0
    x = list(range(0,num_timestep))
    x_loss = list(range(0,num_epoch))

    plt.subplot(2,2,1)
    plt.plot(pred_train_last[:num_timestep, substance_type], marker='+', linewidth=1)
    plt.plot(true_train_last[:num_timestep, substance_type], alpha=0.7, linewidth=1)
    plt.plot(X[:num_timestep, substance_type], '--', linewidth=1)
    plt.legend(['y_pred @ t + {}'.format(1), 'y_true @ t + {}'.format(1), 'x @ t + {}'.format(0)])
    plt.title('Substance type A prediction at {} epoch, Train'.format(num_epoch))

    plt.subplot(2,2,2)
    plt.plot(pred_test[:num_timestep, substance_type], marker='+', linewidth=1)
    plt.plot(Y_test[:num_timestep, substance_type], linewidth=1)
    plt.plot(X_test[:num_timestep, substance_type], '--', linewidth=1)
    plt.legend(['y_pred @ t + {}'.format(1), 'y_true @ t + {}'.format(1), 'x @ t + {}'.format(0)])
    plt.title('Substance type A prediction at {} epoch, Test'.format(num_epoch))

    ##### Plot Training Loss #####
    plt.subplot(2,2,3)
    plt.plot(x_loss, loss_hist)
    plt.title('Training Loss')
    plt.xticks()
    plt.yticks()
    plt.savefig('expt_lorenz/' + optim_name + '/' + str(time_step) + '/' + 'Time Space, Training Loss, Test Loss with ' + 'lr=' + str(lr), format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    plt.show()
    plt.close("all")

    return