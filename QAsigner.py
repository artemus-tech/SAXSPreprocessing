# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt  # http://matplotlib.org
import numpy as np
import cast as ct
import os


class GridGenerator:
    """            	
    Rg - Grid Generation class
 
    """

    def __del__(self):
        """Destructor"""
        print('destroyed')

    def __init__(self, ax, fig, restrict=2):
        """Construtor method, initialize presets"""
        # event definition
        self.event = None
        # set marker size
        self.msz = 8
        # set marker zoome size
        self.marker_zoom_size = 12
        # set picker
        self.pick_numb = 5
        # set numbers of points
        self.restrict = restrict
        # generate  vector abscissa 
        # np.logspace(np.log10(RgMin), np.log10(RgMax), self.point_numb)
        self.src_axis_scale_x = np.empty([1, 1])
        # generate  vector ordinat 
        # np.arange(0, self.point_numb, 1)
        self.src_axis_scale_y = np.empty([1, 1])
        # set path to result grid containing file
        self.res_grid_path = "./result.txt"
        # main point collection storage 
        self.curve_points = []
        self.q1q2 = []
        self.clickId = None
        self.closeId = None
        self.moveId = None
        self.ax = ax

    def run(self):
        # save src-generated Grid   

        plt.ion()
        if len(self.src_axis_scale_x) > 1:
            for i in range(len(self.src_axis_scale_x)):
                point, = plt.plot(self.src_axis_scale_x[i], self.src_axis_scale_y[i], 'bo', markersize=self.msz)
                self.curve_points.append(point)
            np.savetxt(self.res_grid_path, self.src_axis_scale_x.T)

    def connect(self, fig):
        self.clickId = fig.canvas.mpl_connect('button_press_event', self.on_click)
        # connect to click motion_notify_event
        self.moveId = fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        # connect to close plot window event
        self.closeId = fig.canvas.mpl_connect('close_event', self.on_close)

    def disconnect(self, fig):
        if self.clickId:
            fig.canvas.mpl_disconnect(self.clickId)
        if self.moveId:
            fig.canvas.mpl_disconnect(self.moveId)
        if self.closeId:
            fig.canvas.mpl_disconnect(self.closeId)

    def __in_array_q1q2(self, event):
        for point in self.q1q2:
            if (point.contains(event)[0] == True):
                point.remove()
                # trace, to show deleted points formed location
                target, = plt.plot(point.get_xdata()[0], point.get_ydata()[0], 'wo', markersize=self.msz,
                                   picker=self.pick_numb)
                self.q1q2.remove(point)

                target.remove()

        return self.q1q2

    def on_click(self, event):
        #    """append points"""
        #    print("Click")
        #    print(event.canvas.figure.axes)          
        print(self.ax)
        print(event.inaxes)

        if event.inaxes is not None and event.inaxes == self.ax:
            print("in ax")

            print("CLICK")

            axes = event.inaxes

            # for axes in event.canvas.figure.axes[0]:
            # axes = event.canvas.figure.axes
            # to prevent merge modes with adding points procedure
            if axes.get_navigate_mode() is None:

                if event.xdata is not None and event.ydata is not None:
                    # get current axis
                    ax = plt.gca()
                    # overlay plots.
                    # ax.hold(True)
                    # left click branch
                    if event.button == 1 and len(self.q1q2) < self.restrict:
                        point, = plt.plot(event.xdata, event.ydata, 'ro', markersize=self.msz, picker=self.pick_numb)
                        self.q1q2.append(point)
                    # right click branch
                    if event.button == 3:
                        # self.curve_points = self.__in_array(event)
                        self.q1q2 = self.__in_array_q1q2(event)
                    # resfresh plot
                    plt.draw()
                    # save result into file and buffer
                    appended = np.unique(sorted([point.get_xdata()[0] for point in self.q1q2]))
                    np.savetxt("./appended", appended)

    def on_move(self, event):
        if event.inaxes is not None and event.inaxes == self.ax:
            for point in self.q1q2:
                should_be_zoomed = (point.contains(event)[0] == True)
                """onmousemove  return 30, onmouseout return 10"""
                marker_size = 2 * self.msz if should_be_zoomed else self.msz
                # zoom point
                point.set_markersize(marker_size)
            # resfresh plot
            plt.draw()

    def on_close(self, event):
        pass
