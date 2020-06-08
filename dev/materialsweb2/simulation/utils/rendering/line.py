import logging

import simulation
from .point import Point
from simulation.utils.rendering.renderable import *
import matplotlib as plt
logger = logging.getLogger(__name__)

class Line(Renderable):
    def __init__(self, pts, label=None, fill=False, **kwargs):
        if isinstance(pts, Line):
            self = pts
            return
        self.label = label
        self.fill = fill
        self.points = [ Point(pt) for pt in pts ]
        self.options = kwargs

    @property
    def dim(self):
        return self.points[0].dim

    @property
    def as_pairs(self):
        return [ pt.coord for pt in self.points ]

    @property
    def as_axes(self):
        return [ [ pt.coord[i] for pt in self.points ] 
                   for i in range(self.dim) ]

    def draw_in_matplotlib(self, **kwargs):
        if not kwargs.get('axes'):
            axes = plt.gca()
        else:
            axes = kwargs['axes']

        options = dict(self.options)
        axes.plot(*self.as_axes, **options)

    def get_flot_series(self, **kwargs):
        series = {'data': self.as_pairs, 'lines': {'show':True,
            'fill':self.fill}}
        if self.label:
            series['label'] = self.label
        series.update(self.options)
        series.update(kwargs)
        return series

