
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from pandas_utils import pretty_qcut

def format_number(data_value, indx):
    """
    Formats a number to a string with a suffix.

    Parameters
    ----------
    data_value : int
        The number to be formatted.
    indx : int
        The index of the number in the list of numbers.

    Examples
    --------
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # Sample data
    x = [1, 2, 3, 4, 5]
    y = [1000, 2000, 3000, 4000, 5000]


    # Create a plot
    fig, ax = plt.subplots()

    # Set the y-axis tick formatter
    formatter = FuncFormatter(format_number)
    ax.yaxis.set_major_formatter(formatter)

    # Plot the data
    ax.plot(x, y)

    # Show the plot
    plt.show()

    """
    if data_value >= 1_000_000_000:
        formatter = '{:1.1f}B'.format(data_value*0.000_000_001)
    elif data_value >= 10_000_000:
        formatter = '{:1.0f}M'.format(data_value*0.000_001)
    elif data_value >= 1_000_000:
        formatter = '{:1.1f}M'.format(data_value*0.000_001)
    elif data_value >= 10_000:
        formatter = '{:1.0f}K'.format(data_value*0.001)
    elif data_value >= 1_000:
        formatter = '{:1.1f}K'.format(data_value*0.001)
    else:
        formatter = '{:1.0f}'.format(data_value)
    return formatter


def lollyplot(x,y, title=None, figsize=(6,6), xmin=0, **kwargs):
    color = 'green'
    plt.figure(figsize=figsize)
    plt.hlines(y=x, xmin = 0 , xmax = y, color=color)
    plt.plot(y, x, "o", color=color, **kwargs)
    plt.title(title)
    plt.show()


def prepare_frequency_probability(df, col):

    df_plot = df.groupby(col).agg(
        approved=pd.NamedAgg('approved', 'mean'),
        frequency=pd.NamedAgg('approved', 'count'),
    )
    # df_plot['frequency'] = df_plot['frequency'].astype(int)
    return df_plot
    # df_plot['frequency'].sum()


def plot_barh_count(x, title=None, x_label=None, y_label=None, figsize=(10,5), x_in_percentage=False):

    def change_width(ax, new_value) :
        for patch in ax.patches :
            current_width = patch.get_width()
            diff = current_width - new_value
            # we change the bar width
            patch.set_width(new_value)
            # we recenter the bar
            patch.set_y(patch.get_y() + diff * .5)

    font = {'family': 'serif',
        'color':  'gray',
        'weight': 'light',
        'size': 14,
        'style': 'italic',
    }
    plt.figure(figsize=figsize)
    ax=x.plot(kind="barh", color="lightgreen")
    ax.barh(range(len(x)), x.values, align='center', color="lightgreen")
    ax.xaxis.set_ticks_position('top')
    if x_in_percentage:
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0),)
    plt.title(title, loc='left', fontdict=font, pad=30);
    plt.ylabel(None, labelpad=20);


def plot_frequency_probability(df: pd.DataFrame, xlabel:str=None, 
color1:str='lightgrey', color2:str='blue', probability_order:bool=False) -> None:
    """plots a two-axis graph, where the probability of approval is shown as a 
    blue line and the frequency of data on each group is shown as grey bar.

    Args:
        df (_type_): _description_
        xlabel (_type_, optional): _description_. Defaults to None.
        color1 (str, optional): _description_. Defaults to 'lightgrey'.
        color2 (str, optional): _description_. Defaults to 'blue'.
        probability_order (bool, optional): _description_. Defaults to False.
    """
    #First plot
    if probability_order:
        df = df.sort_values('approved')
    ax = df.plot(y='frequency', color=color1, legend=False, kind='bar', rot=0)
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10,integer=True))
    #Second plot
    axes2 = plt.twinx()
    axes2.set_ylabel("Probability of approval")
    axes2.yaxis.label.set_color(color2)
    df.plot(y='approved', color=color2, legend=False, 
        kind='line', ax=axes2, style='.-'
    )
    axes2.yaxis.set_major_formatter(
        ticker.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False)
    ) # transforms ylabel in percentage
    _, top = plt.ylim()
    plt.ylim(0,top*1.1)
    plt.show()


def plot_sweetviz_target(df, var, target, figsize=(10,5), bins=None, round_to=0,
max_categories=None, title=None, x_label=None, y_label=None, ascending=False):

    if bins is not None:
        df[var] = pretty_qcut(df[var], bins=bins, round_to=round_to)
    
    df_plot1 = df[var].value_counts(normalize=True).sort_index(ascending=False)    
    df_plot2 = df.groupby(var)[target].mean().sort_index(ascending=False)

    if max_categories is not None:
        if len(df_plot1) > max_categories:
            categories = df_plot1.sort_values(ascending=False).index[:max_categories].to_list()
            df_plot1 = df_plot1.loc[categories]
            df_plot2 = df_plot2.loc[categories]

    plot_barh_count(df_plot1, title=title, x_in_percentage=True, figsize=figsize, 
        x_label=x_label, y_label=y_label)
    # plt.plot(df_plot2.values[::-1], df_plot2.index.values[::-1], 'o-', color="darkgreen")
    plt.plot(df_plot2.values[::1], np.arange(0,len(df_plot2)), 'o-', color="darkgreen")
    # plt.plot(df_plot2.values, 'o-', color="darkgreen")
    return df_plot1, df_plot2


def generate_save_fig(STUDY_PLOTS):
    # Generate function to save plots in a folder given in STUDY_PLOTS
    def save_fig(name):
        if not os.path.exists(STUDY_PLOTS):
            os.makedirs(STUDY_PLOTS)
        plt.savefig(os.path.join(STUDY_PLOTS, name), bbox_inches = 'tight');        
    return save_fig
