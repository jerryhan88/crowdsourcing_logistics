import os.path as opath
import sys
#
from PyQt5.QtWidgets import QWidget, QApplication, QShortcut
from PyQt5.QtGui import (QPainter, QPen, QColor, QFont, QTextDocument,
                         QImage, QPainterPath, QKeySequence, QPixmap, QCursor)
from PyQt5.QtCore import Qt, QSize, QPoint, QRectF
from colour import Color
from shapely.geometry import Polygon, Point
#
from sgDistrict import get_sgBorder, get_distPoly
from sgLocationPD import pdLoc, get_locationPD
from sgMRT import get_coordMRT, get_mrtNet, get_mrtNetNX, get_route

sgBorder = get_sgBorder()
min_lng, max_lng = 1e400, -1e400
min_lat, max_lat = 1e400, -1e400
for poly in sgBorder:
    for lat, lng in poly:
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
HEIGHT = lat_gap * (WIDTH / lng_gap)

mainFrameOrigin = (60, 100)

SHOW_ALL_PD = True
SHOW_MRT_LINE = False
SHOW_FLOW = True
SHOW_DISTRICT = True


CADI_FLOW = [
    ('Tampines', 'Raffles Place', 1),
    ('Bedok', 'Raffles Place', 1),
    ('Tampines', 'Tanjong Pagar', 1),

    ('Bishan', 'Raffles Place', 1),
    ('Yishun', 'Orchard', 1),
    ('Ang Mo Kio', 'Raffles Place', 1),

    ('Choa Chu Kang', 'Jurong East', 1),
    ('Boon Lay', 'Jurong East', 1),
    ('Yew Tee', 'Jurong East', 1),

    ('Lakeside', 'Tanjong Pagar', 1),
    ('Lakeside', 'Raffles Place', 1),
    ('Boon Lay', 'Tanjong Pagar', 1),
]


def convert_GPS2xy(lng, lat):
    x = (lng - min_lng) / lng_gap * WIDTH
    y = (max_lat - lat) / lat_gap * HEIGHT
    return x, y


class Singapore(object):
    def __init__(self):
        self.sgBoarderXY = []
        for poly in sgBorder:
            sgBorderPartial_xy = []
            for lat, lng in poly:
                x, y = convert_GPS2xy(lng, lat)
                sgBorderPartial_xy += [(x, y)]
            self.sgBoarderXY.append(sgBorderPartial_xy)
        self.sgDistrictXY = {}
        self.sgDistrictPolyXY = {}
        distPoly = get_distPoly()
        for dist_name, poly in distPoly.items():
            points = []
            for lat, lng in poly:
                points.append(convert_GPS2xy(lng, lat))
            self.sgDistrictXY[dist_name] = points
            self.sgDistrictPolyXY[dist_name] = Polygon(points)

    def get_distName(self, x, y):
        p0 = Point(x, y)
        for dist_name, poly in self.sgDistrictPolyXY.items():
            if p0.within(poly):
                return dist_name
        else:
            return None

    def drawPoly(self, qp, poly):
        for i in range(len(poly) - 1):
            x0, y0 = poly[i]
            x1, y1 = poly[i + 1]
            qp.drawLine(x0, y0, x1, y1)
        x0, y0 = poly[len(poly) - 1]
        x1, y1 = poly[0]
        qp.drawLine(x0, y0, x1, y1)

    def draw(self, qp):
        if SHOW_DISTRICT:
            pen = QPen(Qt.black, 0.5, Qt.DashLine)
            qp.setPen(pen)
            for dist_name, poly in self.sgDistrictXY.items():
                self.drawPoly(qp, poly)
        pen = QPen(Qt.black, 1)
        qp.setPen(pen)
        for _, poly in enumerate(self.sgBoarderXY):
            self.drawPoly(qp, poly)


class Station(object):
    STN_markSize = 15

    def __init__(self, STN, lat, lng):
        self.STN = STN
        cx, cy = convert_GPS2xy(lng, lat)
        self.cPoint = QPoint(cx - Station.STN_markSize / 2, cy - Station.STN_markSize / 2)
        self.pixmap = QPixmap('mrtMark.png').scaledToWidth(Station.STN_markSize)

    def draw(self, qp):
        qp.drawPixmap(self.cPoint, self.pixmap)


class LocPD(object):
    pen = QPen(Qt.black, 1, Qt.SolidLine)
    dotSize = 5

    def __init__(self, typePD, lat, lng, nearestMRT, Duration, Location, District):
        self.typePD = typePD
        self.cx, self.cy = convert_GPS2xy(lng, lat)

    def draw(self, qp):
        # qp.setFont(Path.font)
        #
        qp.setPen(LocPD.pen)
        qp.setBrush(Qt.NoBrush)
        self.drawingFunc = qp.drawEllipse if self.typePD == 'P' else qp.drawRect
        self.drawingFunc(self.cx - LocPD.dotSize / 2, self.cy - LocPD.dotSize / 2,
              LocPD.dotSize, LocPD.dotSize)


class Flow(object):
    def __init__(self, count, route, mrt_coords):
        self.count = count
        self.points = []
        for mrt in route:
            lat, lng = mrt_coords[mrt]
            x, y = convert_GPS2xy(lng, lat)
            self.points.append(QPoint(x, y))

    def set_weight(self, sumCount):
        self.weight = self.count / float(sumCount)

    def draw(self, qp):
        for i in range(len(self.points) - 1):
            p0 = self.points[i]
            p1 = self.points[i + 1]
            qp.drawLine(p0, p1)


class Network(object):
    thickness = 1.0
    linePen = {
        'BP': QPen(QColor(Color('grey').get_hex_l()), thickness, Qt.SolidLine),
        'CC': QPen(QColor(Color('orange').get_hex_l()), thickness, Qt.SolidLine),
        'EW': QPen(QColor(Color('green').get_hex_l()), thickness, Qt.SolidLine),
        'NE': QPen(QColor(Color('purple').get_hex_l()), thickness, Qt.SolidLine),
        'NS': QPen(QColor(Color('red').get_hex_l()), thickness, Qt.SolidLine),
        'PTC': QPen(QColor(Color('grey').get_hex_l()), thickness, Qt.SolidLine),
        'STC': QPen(QColor(Color('grey').get_hex_l()), thickness, Qt.SolidLine),
    }

    def __init__(self, mrt_coords):
        mrtNet = get_mrtNet()
        self.lines = []
        for lineName, connections in mrtNet.items():
            for paris in connections:
                edge = []
                for mrt in paris:
                    lat, lng = mrt_coords[mrt]
                    x, y = convert_GPS2xy(lng, lat)
                    p = QPoint(x, y)
                    edge.append(p)
                self.lines.append([lineName, edge])

    def draw(self, qp):
        for lineName, (p0, p1) in self.lines:
            qp.setPen(Network.linePen[lineName])
            qp.drawLine(p0, p1)


class Viz(QWidget):
    font = QFont('Decorative', 15)
    labelH = 30
    unit_labelW = 15

    def __init__(self):
        super().__init__()
        self.sg = Singapore()
        self.objForDrawing = [self.sg]
        #
        if SHOW_ALL_PD:
            self.pdLoc = []
            locationPD = get_locationPD()
            for o in locationPD:
                self.pdLoc.append(LocPD(*o))
            self.objForDrawing += self.pdLoc
        mrt_coords = get_coordMRT()
        if SHOW_MRT_LINE:
            self.mrts = []
            for STN, (lat, lng) in mrt_coords.items():
                self.mrts.append(Station(STN, lat, lng))
            self.objForDrawing += self.mrts
            #
            self.mrtNetwork = Network(mrt_coords)
            self.objForDrawing += [self.mrtNetwork]
        if SHOW_FLOW:
            flows = []
            mrtNetNX = get_mrtNetNX()
            for mrt0, mrt1, count in CADI_FLOW:
                route = get_route(mrtNetNX, mrt0, mrt1)
                flows.append(Flow(count, route, mrt_coords))
            self.objForDrawing += flows
        #
        self.mousePressed = False
        self.px, self.py = -1, -1
        #
        self.initUI()
        #
        self.shortcut = QShortcut(QKeySequence('Ctrl+W'), self)
        self.shortcut.activated.connect(self.close)

    def initUI(self):
        self.setGeometry(mainFrameOrigin[0], mainFrameOrigin[1], WIDTH, HEIGHT)
        self.setWindowTitle('Viz')
        self.setFixedSize(QSize(WIDTH, HEIGHT))
        self.show()

    def mousePressEvent(self, QMouseEvent):
        if self.mousePressed:
            self.mousePressed = False
            self.px, self.py = -1, -1
            self.update()
        else:
            pos = QMouseEvent.pos()
            x, y = [f() for f in [pos.x, pos.y]]
            dist_name = self.sg.get_distName(x, y)
            if dist_name:
                self.mousePressed = True
                self.selDistLabel = QTextDocument()
                self.selDistLabel.setHtml(dist_name)
                self.selDistLabel.setDefaultFont(Viz.font)
                self.labelW = len(dist_name) * Viz.unit_labelW
                self.px, self.py = x - self.labelW / 2, y - Viz.labelH
                #
                print(dist_name)
                self.update()

    # def mouseReleaseEvent(self, QMouseEvent):
    #     cursor = QCursor()
        # print(cursor.pos())



    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.drawCanvas(qp)
        qp.end()

    def drawCanvas(self, qp):
        for o in self.objForDrawing:
            o.draw(qp)
        if self.mousePressed:
            qp.translate(self.px, self.py)
            self.selDistLabel.drawContents(qp, QRectF(0, 0, self.labelW, Viz.labelH))
            qp.translate(-self.px, -self.py)



if __name__ == '__main__':
    # get_sgBoarderXY()

    app = QApplication(sys.argv)
    viz = Viz()
    sys.exit(app.exec_())