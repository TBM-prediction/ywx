def plots(df_raw, cols=5, figsize=5):
    num_plots = df_raw.shape[1] - 2 # omit index and timestamp
    rows = num_plots // cols + 1
    figsize = [figsize * o for o in (cols, rows)]

    axes = df_raw.plot(y=list(range(num_plots)), subplots=True, figsize=figsize, layout=(rows, cols))

    # use chinese fonts
    from matplotlib import font_manager as fm
    prop = fm.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc')
    for ax in axes.flatten():
        legend = ax.legend()
        plt.setp(legend.texts, fontproperties=prop)
        xlabel = ax.xaxis.label
        plt.setp(xlabel, fontproperties=prop)

    plt.tight_layout()
    return axes