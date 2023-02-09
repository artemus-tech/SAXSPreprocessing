# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import cast as ct
from pltconf import *
from assets import *
from QAsigner import GridGenerator
from matplotlib.widgets import Button

handlers = []


def plot_handler(func):
    handlers.append(func)
    return func


def plot_clear(plotfunc):
    # this is the key line. There's the aditional self parameter
    # def wrap(*args, **kwargs):
    def wrap(self, *args, **kwargs):
        # you can use self here as if you were inside the class
        plt.cla()
        plotfunc(self, *args, **kwargs)
        # fig.canvas.draw_idle()
        fig.canvas.draw()

    return wrap


def plot_clear(plotfunc):
    # this is the key line. There's the additional self parameter
    def wrap(self, *args, **kwargs):
        # you can use self here as if you were inside the class
        plt.cla()
        plotfunc(self, *args, **kwargs)
        # fig.canvas.draw_idle()
        fig.canvas.draw()

    return wrap


class Index:
    ind = 0

    def __init__(self):
        plt.sca(ax)
        # plt.sca(ax)
        # plt.figure(1)
        print(handlers)

    @plot_clear
    def next(self, event):
        self.ind += 1
        hr = handlers[self.ind]()
        # xdata = hr[0]
        # ydata = hr[1]

        # ax.plot(xdata,ydata)

        # print(handlers[self.ind])
        # plt.clf()
        # l.set_ydata(ydata)
        # l.set_xdata(xdata)
        # plt.draw()
        fig.canvas.draw_idle()

    @plot_clear
    def prev(self, event):
        self.ind -= 1
        # plt.clf()

        print(handlers[self.ind])

        hr = handlers[self.ind]()
        # xdata = hr[0]
        # ydata = hr[1]

        # l.set_ydata(ydata)
        # l.set_xdata(xdata)
        # ax.plot(xdata,ydata)
        # plt.draw()


def get_src_indicatrix():
    Indicatrix = np.loadtxt(curvePath)
    src_rows, src_cols = np.shape(Indicatrix)
    monitor_matrix(Indicatrix, "Indicatrix")
    # SRC INDICATRIX COLUMN
    q_SRC = ct.scat_vect(Indicatrix[:, 0])
    I_SRC = Indicatrix[:, 1]
    dI_SRC = np.zeros(src_rows)
    monitor_matrix(q_SRC, "qsrc")
    return [q_SRC, I_SRC, dI_SRC]


@plot_handler
def plot_src_indicatrix():
    plt.suptitle("Source Indicatrix")
    ax.set_ylabel("$I(q), arb. units$")
    ax.set_xlabel("$q, nm^{-1}$")
    Indicatrix = get_src_indicatrix()
    ax.plot(q_SRC, I_SRC)
    return Indicatrix


@plot_handler
def plot_indicatrix_separator():
    gg = data["ggq1q2"]
    gg.src_axis_scale_x = q_SRC
    gg.src_axis_scale_y = I_SRC
    gg.res_grid_path = "./qselected"
    gg.q1q2 = []
    gg.run()
    # connect to click button_press_event
    gg.connect(fig)
    # super title for entire plot
    plt.suptitle("SELECT Q1,Q2 values from the range the closest to linear")
    ax.set_ylabel("$I(q), arb. units$")
    ax.set_xlabel("$q, nm^{-1}$")
    # set double log scale
    plt.loglog()


@plot_handler
def plot_indicatrix_without_slope():
    plt.suptitle("Scatteing Intens without linear regression slope constant")

    gg = data["ggq1q2"]
    q1, q2 = [x.get_xdata()[0] for x in gg.q1q2]
    gg.disconnect(fig)

    #############INITIALIZE LINEAR CURVES########
    # vq1 = np.full( (src_rows, ), q1, dtype=float)
    # vq2 = np.full( (src_rows, ), q2, dtype=float)
    #####################################################
    # plot_multiple_func(plt, False, (q_SRC, I_SRC,"SOURCE"), (vq2, I_SRC,"q2"),(vq1, I_SRC,"q1"))
    # plot_multiple_func(plt, True, (q_SRC, I_SRC,"SOURCE"), (vq2, I_SRC,"q2"),(vq1, I_SRC,"q1"))

    # CUT OF MAIN CURVE
    M = qIdI_term(q_SRC, I_SRC, dI_SRC, q1, q2)
    q_EXTACT1 = M[:, 0]
    I_EXTACT1 = M[:, 1]
    #############POWER RISE UP TO 4 for Cuinier#########################
    q_EXTACT1_res4 = np.power(M[:, 0], 4)
    I_EXTACT1_res4 = np.power(M[:, 0], 4) * M[:, 1]

    print(q1, q2)

    q14 = np.power(q1, 4)
    q24 = np.power(q2, 4)

    M4 = qIdI_term(q_EXTACT1_res4, I_EXTACT1_res4, I_EXTACT1_res4, q14, q24)
    lr = linregress(q_EXTACT1_res4, I_EXTACT1_res4)
    LIg = lr.slope + lr.intercept / q_EXTACT1_res4

    data["I_WithOut_Slope"] = np.array([])

    ########PLOT SOURCE INTENSITY###############
    # plot_two_single_func(plt, q_EXTACT1, LIg, q_EXTACT1, I_EXTACT1)
    ##############################################
    I_WithOut_Slope = I_SRC - lr.slope
    data["I_WithOut_Slope"] = I_WithOut_Slope

    monitor_matrix(data["I_WithOut_Slope"], "I_WithOut_Slope")
    ######################PLOT DEVIATION##############################################
    plot_multiple_func(plt, ("q, nm^{-1}", "I, arb. units"), False, True, False, (q_SRC, I_SRC, "SOURCE"),
                       (q_SRC, I_WithOut_Slope, "I_WithOut_Slope"))
    # plot_multiple_func(plt, ("q, nm^{-1}","I, arb. units"),True, False, False, (q_SRC, I_SRC,"SOURCE"), (q_SRC, I_WithOut_Slope,"I_WithOut_Slope"))
    ######################PLOT DEVIATION LOG##############################################

    # return [q_SRC,I_SRC]
    """
    q1,q2 = [x.get_xdata()[0] for x in gg.q1q2]
    M = qIdI_term(q_SRC,I_SRC, dI_SRC, q1, q2) 
    q_EXTACT1 = M[:,0]
    I_EXTACT1 = M[:,1]
    return [q_EXTACT1, I_EXTACT1]
    #show the plot
    #plt.show()
    """


@plot_handler
def gridgenerator_return():
    plt.suptitle("SELECT Q3,Q4 values from the starting range")
    ax.set_ylabel("$I(q), arb. units$")
    ax.set_xlabel("$q, nm^{-1}$")

    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ##plt.yscale('linear')
    # plt.xscale('linear')
    monitor_matrix(data["I_WithOut_Slope"], "GG I_WithOut_Slope")
    I_WithOut_Slope = data["I_WithOut_Slope"]
    gg = data["ggq3q4"]

    gg.q1q2 = []

    gg.src_axis_scale_x = np.power(q_SRC, 2)
    gg.src_axis_scale_y = np.log(data["I_WithOut_Slope"])

    gg.res_grid_path = "./qselected2.txt"
    gg.run()
    # connect to click button_press_event

    gg.connect(fig)
    # connect to click button_press_event
    plt.title = "Q4,Q5"
    plt.ylabel("$I(q), arb. units$")
    plt.xlabel("$q, nm^{-1}$")
    # plot_multiple_func(plt, ("q, nm^{-1}","I, arb. units"),False,True,False, (q_SRC, I_SRC,"SOURCE"), (q_SRC, I_WithOut_Slope,"I_WithOut_Slope"))

    plot_multiple_func(plt, ("q, nm^{-1}", "I, arb. units"), True, False, False, (q_SRC, I_SRC, "SOURCE"),
                       (q_SRC, I_WithOut_Slope, "I_WithOut_Slope"))


@plot_handler
def result_step():
    gg = data["ggq3q4"]

    I_WithOut_Slope = data["I_WithOut_Slope"]

    q4, q5 = [x.get_xdata()[0] for x in gg.q1q2]
    q1, q2 = [x.get_xdata()[0] for x in data["ggq1q2"].q1q2]

    M3 = qIdI_term(q_SRC, I_WithOut_Slope, dI_SRC, q4, q5)
    I_EXTRACT3 = M3[:, 1]
    q_EXTRACT3 = M3[:, 0]
    ################################

    D = get_D(q_SRC, I_SRC)

    r0 = get_r0(q1)

    KSI = get_KSI(r0)

    print(
        "==============================================START D|ro|KSI====================================================")
    print(D)
    print(r0)
    print(KSI)
    print(
        "==============================================END D|ro|KSI====================================================")

    ################################
    SF = SFactor(q_EXTRACT3[0], D, KSI, r0)
    MN = get_MN(I_EXTRACT3, SF)
    M4 = qIdI_term(np.power(q_SRC, 2), np.log(I_WithOut_Slope), np.log(I_WithOut_Slope), q6, q7)
    q_EXTRACT4 = M4[:, 0]
    I_EXTRACT4 = M4[:, 1]
    a3 = np.log(MN)

    DELTA1 = get_delta(I_EXTRACT3, q_EXTRACT3, a3, -6, D, KSI, r0)
    DELTA2 = get_delta(I_EXTRACT3, q_EXTRACT3, DELTA_a3, DELTA_b3, DELTA_D, DELTA_KSI, DELTA_r0)
    mm = min_get_delta(I_EXTRACT3, q_EXTRACT3, [DELTA_a3, DELTA_b3, DELTA_D, DELTA_KSI, DELTA_r0])
    opt_a3, opt_b3, opt_D, opt_KSI, opt_r0 = mm.x

    print(
        "==============================================START opt_SF====================================================")
    opt_SF = SFactor(q_EXTRACT3[0], opt_D, opt_KSI, opt_r0)
    print(opt_SF)
    print(
        "==============================================END opt_SF====================================================")

    print(
        "==============================================START INPUT====================================================")
    opt_MN = get_MN(I_EXTRACT3, opt_SF)
    print("==============================================END INPUT====================================================")
    print(opt_a3, opt_b3, opt_D, opt_KSI, opt_r0)

    plt.plot(q_EXTRACT3, I_EXTRACT3, ms=2, marker="_", color="b")
    plt.plot(q_EXTRACT3, I_EXTRACT3 / SFactor(q_EXTRACT3, opt_D, opt_KSI, opt_r0), ms=1, marker="_", color="r")
    plt.plot(q_EXTRACT3, opt_MN * SFactor(q_EXTRACT3, opt_D, opt_KSI, opt_r0), ms=3, marker="o", color="g")

    plt.loglog()

    print(
        "==============================================START INPUT====================================================")
    print(q_EXTRACT3, D, KSI, r0, MN)

    print("==============================================END INPUT====================================================")

    print(
        "==============================================Start SPFactor for corresponding params====================================================")
    print(SFactor(q_EXTRACT3, D, KSI, r0) * MN)
    print(
        "==============================================END SPFactor for corresponding params====================================================")

    # plot_multiple_func(plt, ("q, nm^{-1}", "arb. units"), False,True, True, (q_SRC, I_SRC, "I_SOURCE"),(q_EXTRACT3, I_EXTRACT3,"I_EXTRACT3"),(q_EXTRACT3, SFactor(q_EXTRACT3, D, KSI, r0)* MN,"SFactor"))


from matplotlib.backend_tools import ToolBase
class NewTool(ToolBase):
    image = r"C:\Windows\Web\Wallpaper\Theme1\img3.jpg"

if __name__ == '__main__':
    fig, ax = plt.subplots()

    fig.subplots_adjust(bottom=0.2)
    axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
    axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    q_SRC, I_SRC, dI_SRC = get_src_indicatrix()

    data = {
        "I_WithOut_Slope": np.array([]),
        "ggq1q2": GridGenerator(ax, fig),
        "ggq3q4": GridGenerator(ax, fig),
        "q_SRC": q_SRC,
        "I_SRC": I_SRC,
        "dI_SRC": dI_SRC
    }

    handlers[0]()
   
    callback = Index()

    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    plt.show()
