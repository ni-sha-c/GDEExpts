import numpy as np
import matplotlib.pyplot as plt

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
test_optim_plot("lorenz")