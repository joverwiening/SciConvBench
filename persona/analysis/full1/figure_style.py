# Figure style loaded by the plotting scripts.
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'lines.linewidth': 1.5, 'lines.markersize': 5,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'axes.titlepad': 8, 'axes.labelpad': 4,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.major.size': 4, 'ytick.major.size': 4,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'legend.frameon': False, 'legend.borderpad': 0.3, 'legend.handlelength': 1.5,
})

COLORS = {
    'primary': '#2563EB', 'secondary': '#DC2626', 'tertiary': '#059669', 'quaternary': '#D97706',
    'quinary': '#7C3AED', 'senary': '#DB2777', 'gray': '#6B7280', 'light_gray': '#D1D5DB',
}
COLOR_CYCLE = list(COLORS.values())[:6]
plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=COLOR_CYCLE)

PRESENTATION = {'font.size': 12, 'axes.labelsize': 13, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
                'legend.fontsize': 11, 'lines.linewidth': 2.0}


def figsize(width=4.5, aspect=0.75):
    return (width, width * aspect)


def save_fig(fig, name, directory='plots'):
    import os
    os.makedirs(directory, exist_ok=True)
    for ext in ['png', 'pdf']:
        path = os.path.join(directory, f'{name}.{ext}')
        fig.savefig(path)
        print(f'Saved {path}')
    plt.close(fig)
