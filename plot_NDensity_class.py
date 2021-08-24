import NDensity as nd
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

dc_files = glob.glob("*_dc.dat")
dc_files.sort()

start_sys = 0
end_sys = 14

files_to_plot = dc_files[start_sys:end_sys]

for num in range(len(files_to_plot)):
    system = files_to_plot[num]
    dp = nd.density_profile(file_name=system)

    if num < 7:
        color = 'Red'
        if num == 0:
            dp.plot_class(solo_fig=True,
                          min_color=-4,
                          color=color,
                          alpha=0.3)
        else:
            dp.plot_class(solo_fig=False,
                          min_color=-4,
                          color=color,
                          alpha=0.3)
    elif num >= 7 and num < 11:
        color = 'Blue'
        dp.plot_class(solo_fig=False,
                      min_color=-4,
                      color=color,
                      alpha=0.3,
                      hatch='/')
    elif num >= 11 and num < 13:
        color = 'Gray'
        dp.plot_class(solo_fig=False,
                      min_color=-4,
                      color=color,
                      alpha=0.3,
                      hatch='+')
    elif num >= 13:
        color = 'Magenta'
        dp.plot_class(solo_fig=False,
                      min_color=-12,
                      color=color,
                      alpha=0.3,
                      hatch='.')


plt.ylim(0.925, 5)
plt.xlim(-0.7, 1.8)
plt.title("Binary Progenitors", fontsize=20)
plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2)

UCXBs = mpatches.Patch(color='Red', alpha=0.3, label='UCXBs')
Short = mpatches.Patch(color='Blue', hatch='/',
                       alpha=0.3, label='Short Period')
Medium = mpatches.Patch(color='gray', hatch='+',
                        alpha=0.3, label='Medium Period')
Long = mpatches.Patch(color='Magenta', hatch='.',
                      alpha=0.3, label='Long Period')
plt.legend(handles=[UCXBs, Short, Medium, Long], loc=4)

plt.savefig("Progen_Class.pdf")
