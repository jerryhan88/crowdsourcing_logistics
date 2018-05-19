import os.path as opath
import sys
#
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import (QPen, QColor, QFont, QTextDocument, QPainter, QImage, QPainterPath)
from PyQt5.QtCore import Qt, QSize
#
from sgDistrict import get_sgBorder
from vizEuclidean import sort_clockwise

sgBorder = get_sgBorder()
min_lng, max_lng = 1e400, -1e400
min_lat, max_lat = 1e400, -1e400
for poly in sgBorder:
    for lng, lat in poly:
        if lng < min_lng:
            min_lng = lng
        if lng > max_lng:
            max_lng = lng
        if lat < min_lat:
            min_lat = lat
        if lat > max_lat:
            max_lat = lat

lng_gap = max_lng - min_lng
lat_gap = max_lat - min_lat

WIDTH = 1800.0

SCALE = WIDTH / lat_gap
HEIGHT = lng_gap * SCALE

mainFrameOrigin = (60, 100)


def get_sgBoarderXY():
    sgBorder_xy = []
    for poly in sgBorder:
        sgBorderPartial_xy = []
        for lat, lng in poly:
            x, y = convert_GPS2xy(lng, lat)
            sgBorderPartial_xy += [(x, y)]
        sgBorder_xy.append(sgBorderPartial_xy)
    return sgBorder_xy


def convert_GPS2xy(lng, lat):
    x = (lng - min_lng) * SCALE
    y = (max_lat - (lat - min_lat)) * SCALE
    return x, y


class Viz(QWidget):
    def __init__(self):
        super().__init__()
        self.sgBorder = []

        _sgBoarderXY = get_sgBoarderXY()

        min_x, min_y = 1e400, 1e400
        for poly in _sgBoarderXY:
            for x, y in poly:
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y

        self.sgBoarderXY = []
        for poly in _sgBoarderXY:
            newPoly = []
            for x, y in poly:
                newPoly.append([x - min_x, y - min_y])
            self.sgBoarderXY.append(newPoly)

        self.initUI()

    def initUI(self):
        self.setGeometry(mainFrameOrigin[0], mainFrameOrigin[1], WIDTH, HEIGHT)
        self.setWindowTitle('Viz')
        self.setFixedSize(QSize(WIDTH, HEIGHT))
        self.show()

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.drawCanvas(qp)
        qp.end()

    def drawCanvas(self, qp):
        pen = QPen(Qt.black, 1)
        qp.setPen(pen)
        for j, poly in enumerate(self.sgBoarderXY):
            # if j != 2:
            #     continue
            for i in range(len(poly) - 1):
                x0, y0 = poly[i]
                x1, y1 = poly[i + 1]
                qp.drawLine(x0, y0, x1, y1)
            x0, y0 = poly[len(poly) - 1]
            x1, y1 = poly[0]
            qp.drawLine(x0, y0, x1, y1)






if __name__ == '__main__':
    # get_sgBoarderXY()

    app = QApplication(sys.argv)
    viz = Viz()
    sys.exit(app.exec_())