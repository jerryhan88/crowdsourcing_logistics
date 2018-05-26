import os.path as opath
import sys
import pickle
from itertools import chain
import numpy as np
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

FRAME_ORIGIN = (60, 100)

SHOW_ALL_PD = False
SHOW_MRT_LINE = False
SHOW_DISTRICT = True

SAVE_IMAGE = True

pallet = [
    Color('blue').get_hex_l(),
    Color('brown').get_hex_l(),
    Color('magenta').get_hex_l(),
    Color('green').get_hex_l(),
    Color('indigo').get_hex_l(),
    Color('khaki').get_hex_l(),
    Color('maroon').get_hex_l(),
    Color('navy').get_hex_l(),
    Color('orange').get_hex_l(),
    Color('pink').get_hex_l(),
    Color('red').get_hex_l(),
    Color('grey').get_hex_l(),
]

D360 = 360.0


def sort_clockwise(points):
    c = np.array(list(map(sum, zip(*points)))) / len(points)
    cp = c + np.array([1, 0])
    clockwiseDegrees = []
    for i, ca in enumerate([np.array(p) - c for p in points]):
        degree = np.degrees(np.arccos(np.dot(ca, cp) / (np.linalg.norm(ca) * np.linalg.norm(cp))))
        if 0 <= ca[1]:
            clockwiseDegrees.append([degree, points[i]])
        else:
            clockwiseDegrees.append([D360 - degree, points[i]])
    return [p for _, p in sorted(clockwiseDegrees)]


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
    STN_markSize = 25

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
    lineProp = 30

    def __init__(self, weight, route, mrt_coords):
        self.weight = weight
        self.points = []
        for mrt in route:
            lat, lng = mrt_coords[mrt]
            x, y = convert_GPS2xy(lng, lat)
            self.points.append(QPoint(x, y))

    def draw(self, qp):
        pen = QPen(Qt.black, self.weight * Flow.lineProp, Qt.SolidLine)
        qp.setPen(pen)
        for i in range(len(self.points) - 1):
            p0 = self.points[i]
            p1 = self.points[i + 1]
            qp.drawLine(p0, p1)


class Network(object):
    thickness = 1.0
    lineStyle = Qt.DotLine
    linePen = {
        'BP': QPen(QColor(Color('grey').get_hex_l()), thickness, lineStyle),
        'CC': QPen(QColor(Color('orange').get_hex_l()), thickness, lineStyle),
        'EW': QPen(QColor(Color('green').get_hex_l()), thickness, lineStyle),
        'NE': QPen(QColor(Color('purple').get_hex_l()), thickness, lineStyle),
        'NS': QPen(QColor(Color('red').get_hex_l()), thickness, lineStyle),
        'PTC': QPen(QColor(Color('grey').get_hex_l()), thickness, lineStyle),
        'STC': QPen(QColor(Color('grey').get_hex_l()), thickness, lineStyle),
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


class Bundle(object):
    font = QFont('Decorative', 15)
    labelH = 30
    unit_labelW = 15

    def __init__(self, bid, assTaskIDs, points):
        self.bid = bid
        #
        self.label = QTextDocument()
        _assTaskIDs = str(assTaskIDs)
        self.label.setHtml("b<sub>%d</sub>: %s" % (self.bid, _assTaskIDs))
        self.label.setDefaultFont(Bundle.font)
        #
        self.points = points
        c = np.array(list(map(sum, zip(*points)))) / len(points)
        self.labelW = (len("bx: ") + len(_assTaskIDs)) * Bundle.unit_labelW
        self.lx, self.ly = c[0], c[1]

    def draw(self, qp):
        # qp.setFont(Bundle.font)
        self.drawLabel(qp, self.label,
                  self.lx, self.ly, self.labelW, Bundle.labelH)
        pen = QPen(QColor(pallet[self.bid % len(pallet)]), 3, Qt.DashDotDotLine)
        qp.setPen(pen)
        for i in range(len(self.points) - 1):
            x0, y0 = self.points[i]
            x1, y1 = self.points[i + 1]
            qp.drawLine(x0, y0, x1, y1)
        x0, y0 = self.points[len(self.points) - 1]
        x1, y1 = self.points[0]
        qp.drawLine(x0, y0, x1, y1)

    def drawLabel(self, qp, label, cx, cy, w, h):
        qp.translate(cx - w / 2, cy - h / 2)
        label.drawContents(qp, QRectF(0, 0, w, h))
        qp.translate(-(cx - w / 2), -(cy - h / 2))


class Viz(QWidget):
    font = QFont('Decorative', 15)
    labelH = 30
    unit_labelW = 15

    def __init__(self, pkl_files):
        super().__init__()
        self.app_name = 'Viz'
        #
        locationPD = get_locationPD()
        mrt_coords = get_coordMRT()
        #
        self.sg = Singapore()
        self.objForDrawing = [self.sg]
        #
        if SHOW_ALL_PD:
            for o in locationPD:
                self.objForDrawing.append(LocPD(*o))
        if SHOW_MRT_LINE:
            for STN, (lat, lng) in mrt_coords.items():
                self.objForDrawing.append(Station(STN, lat, lng))
            self.objForDrawing.append(Network(mrt_coords))
        if pkl_files:
            drawingInfo = {}
            for k, fpath in pkl_files.items():
                with open(fpath, 'rb') as fp:
                    drawingInfo[k] = pickle.load(fp)
            self.app_name += '-%s' % drawingInfo['prmts']['problemName']
            flow_oridest, task_ppdp = drawingInfo['dplym']
            w_k = drawingInfo['prmts']['w_k']
            mrtNetNX = get_mrtNetNX()
            for k, (mrt0, mrt1) in enumerate(flow_oridest):
                route = get_route(mrtNetNX, mrt0, mrt1)
                self.objForDrawing.append(Flow(w_k[k], route, mrt_coords))
                #
                lat0, lng0 = mrt_coords[mrt0]
                self.objForDrawing.append(Station(mrt0, lat0, lng0))
                lat1, lng1 = mrt_coords[mrt1]
                self.objForDrawing.append(Station(mrt1, lat1, lng1))
            #
            ln_locO = {o.Location: o for o in locationPD}
            task_pdO = []
            for pLoc, dLoc in task_ppdp:
                po, do = LocPD(*ln_locO[pLoc]), LocPD(*ln_locO[dLoc])
                self.objForDrawing.append(po)
                self.objForDrawing.append(do)
                task_pdO.append([po, do])
            #
            if 'CWL' in pkl_files['sol']:
                C, q_c = [drawingInfo['sol'].get(k) for k in ['C', 'q_c']]
                generatedBundles = [C[c] for c in range(len(C)) if q_c[c] > 0.5]
                for bid, bc in enumerate(generatedBundles):
                    points = []
                    for tid in bc:
                        po, do = task_pdO[tid]
                        points.append([po.cx, po.cy])
                        points.append([do.cx, do.cy])
                    points = sort_clockwise(points)
                    self.objForDrawing.append(Bundle(bid, bc, points))
        #
        self.mousePressed = False
        self.px, self.py = -1, -1
        #
        self.initUI()
        #
        self.shortcut = QShortcut(QKeySequence('Ctrl+W'), self)
        self.shortcut.activated.connect(self.close)

    def initUI(self):
        self.setGeometry(FRAME_ORIGIN[0], FRAME_ORIGIN[1], WIDTH, HEIGHT)
        self.setWindowTitle(self.app_name)
        self.setFixedSize(QSize(WIDTH, HEIGHT))
        #
        if SAVE_IMAGE:
            self.image = QImage(WIDTH, HEIGHT, QImage.Format_RGB32)
            self.path = QPainterPath()
            self.image.fill(Qt.white)  ## switch it to else
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
        if SAVE_IMAGE:
            qp = QPainter()
            qp.begin(self.image)
            self.drawCanvas(qp)
            qp.end()

    def save_img(self, img_fpath):
        self.image.save(img_fpath, 'png')

    def drawCanvas(self, qp):
        for o in self.objForDrawing:
            o.draw(qp)
        if self.mousePressed:
            qp.translate(self.px, self.py)
            self.selDistLabel.drawContents(qp, QRectF(0, 0, self.labelW, Viz.labelH))
            qp.translate(-self.px, -self.py)



if __name__ == '__main__':
    import os.path as opath

    prefix = '5out-nt100-dPD30-dp25-fp75'
    pkl_files = {
        'dplym': opath.join('_temp', 'dplym_%s.pkl' % prefix),
        'prmts': opath.join('_temp', 'prmts_%s.pkl' % prefix),
        'sol': opath.join('_temp', 'sol_%s_CWL.pkl' % prefix)
    }

    app = QApplication(sys.argv)
    viz = Viz(pkl_files)
    if SAVE_IMAGE:
        viz.save_img('temp.png')
    sys.exit(app.exec_())
