from utils.imports import *

def set_font(label, fname='/usr/share/fonts/wqy-microhei/wqy-microhei.ttc'):
    # use chinese fonts
    from matplotlib import font_manager as fm
    prop = fm.FontProperties(fname=fname)
    plt.setp(label, fontproperties=prop)

def set_ax_font(ax, fname='/usr/share/fonts/wqy-microhei/wqy-microhei.ttc'):
    if ax.legend_ is not None: set_font(ax.legend_.texts, fname=fname)
    set_font(ax.title, fname=fname)
    if hasattr(ax,'xaxis'): set_font(ax.xaxis.label, fname=fname)
    if hasattr(ax,'yaxis'): set_font(ax.yaxis.label, fname=fname)
    for ticklabels in (ax.xaxis.get_ticklabels(),ax.yaxis.get_ticklabels()):
        if ticklabels:
            for ticklabel in ticklabels: 
                set_font(ticklabel, fname=fname)

def ceildiv(a, b):
    return -(-a//b)

def plots(df_raw, cols=3, unit_figsize=(8, 3), ax=None,
        fname='/usr/share/fonts/wqy-microhei/wqy-microhei.ttc', title=None):
    num_plots = df_raw.shape[1]
    rows = ceildiv(num_plots, cols)
    figsize = (unit_figsize[0] * cols, unit_figsize[1] * rows)

    if ax is not None: ax = ax.flatten()[:num_plots]
    axs = df_raw.plot(subplots=True, figsize=figsize, 
            layout=(rows, cols), ax=ax, legend=ax is None, title=title)

    for ax in axs.flatten():
        set_ax_font(ax)

    plt.tight_layout()
    return axs

def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)

