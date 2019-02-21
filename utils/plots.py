from utils.imports import *

def set_font(label, fname='/usr/share/fonts/wqy-microhei/wqy-microhei.ttc'):
    # use chinese fonts
    from matplotlib import font_manager as fm
    prop = fm.FontProperties(fname=fname)
    plt.setp(label, fontproperties=prop)


def ceildiv(a, b):
    return -(-a//b)

def plots(df_raw, cols=3, unit_figsize=(8, 3), ax=None,
        fname='/usr/share/fonts/wqy-microhei/wqy-microhei.ttc', title=None):
    num_plots = df_raw.shape[1]
    rows = ceildiv(num_plots, cols)
    figsize = (unit_figsize[0] * cols, unit_figsize[1] * rows)

    if ax is not None: ax = ax.flatten()[:num_plots]
    ax = df_raw.plot(subplots=True, figsize=figsize, 
            layout=(rows, cols), ax=ax, legend=ax is None, title=title)

    for a in ax.flatten():
        if a.legend_: set_font(a.legend_.texts, fname=fname)
        if a.xaxis: set_font(a.xaxis.label, fname=fname)

    plt.tight_layout()
    return ax

def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)
