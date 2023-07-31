import numpy as np
import matplotlib.pyplot as plt

def plot_loss():
    training1 = np.loadtxt("./"+ "expt_lorenz" +"/AdamW/0.01/training_loss.csv", delimiter=",", dtype=float)
    training2 = np.loadtxt("./"+ "expt_lorenz" +"/AdamW/0.005/training_loss.csv", delimiter=",", dtype=float)
    training3 = np.loadtxt("./"+ "expt_lorenz" +"/AdamW/0.0005/[0,40] | 18000/training_loss.csv", delimiter=",", dtype=float)

    plt.figure(figsize=(40,10))
    #plt.plot(training1)
    plt.plot(training2)
    plt.plot(training3)

    #plt.legend(['1e-2', '5e-3', '5e-4'])
    plt.legend(['5e-3', '5e-4'])
    plt.title('Training loss vs Epoch with Different Timestep Size')
    plt.xticks()
    plt.yticks()
    plt.savefig('./loss_comparison_node/'+ 'training_loss_comparison_two', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    plt.show()
    plt.close()

def test_optim_plot(system):

    if system == "brusselator":
        folder_name = "expt_brusselator"
    elif system == "sin":
        folder_name = "expt_sin_second_order"
    elif system == "lorenz":
        folder_name = "expt_lorenz"

    ##### load training_loss_csv ######
    Adam_plot = np.loadtxt("./"+ folder_name +"/Adam/training_loss.csv",
                    delimiter=",", dtype=float)

    AdamW_plot = np.loadtxt("./"+ folder_name + "/AdamW/training_loss.csv",
                    delimiter=",", dtype=float)

    GD_plot = np.loadtxt("./" + folder_name + "/Gradient Descent/training_loss.csv",
                    delimiter=",", dtype=float)

    RMSprop_plot = np.loadtxt("./" + folder_name + "/RMSprop/training_loss.csv",
                    delimiter=",", dtype=float)


    ##### create plot and save ######
    plt.figure(figsize=(40,10))
    plt.plot(Adam_plot)
    plt.plot(AdamW_plot)
    plt.plot(GD_plot)
    plt.plot(RMSprop_plot)
    plt.legend(['Adam', 'AdamW', 'Gradient Descent', 'RMSprop'])
    plt.title('NeuralODE Training loss based on different optimizer at lr=5e-4')
    plt.xticks()
    plt.yticks()
    plt.savefig('./loss_comparison_node/'+ system + '_training_loss_0.0005', format='png', dpi=400, bbox_inches ='tight', pad_inches = 0.1)
    plt.show()
    plt.close()

##### run test_optim_plot() #####
#test_optim_plot("lorenz")
plot_loss()