from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def plot_curves(output_y, s, sensitive_feature, name_plot):

    plt.clf()

    y_pre_1 = output_y[s.flatten() == 1]
    y_pre_0 = output_y[s.flatten() == 0]

    sns.set_style("white")
    sns.set_palette(None)

    # Plot the distributions with specified colors
    sns.kdeplot(y_pre_1.squeeze(), label='s==1', color='blue')
    sns.kdeplot(y_pre_0.squeeze(), label='s==0', color='orange')


    # Add labels and title
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.title(f'Probability Distribution of {sensitive_feature} = 0 and = 1.')

    # Set x-axis limits
    plt.xlim(0, 1)

    # Add a legend
    plt.legend()

    # Show the plot
    print("saving ", name_plot)
    plt.savefig(name_plot, format="pdf", bbox_inches="tight")

    #plt.show()

    