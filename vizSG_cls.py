import numpy as np
from shapely.geometry import Polygon, Point

from PyQt5.QtGui import (QFont, QPen, QColor,
                         QTextDocument, QPixmap)
from PyQt5.QtCore import (Qt,
                          QPoint, QRectF)
from colour import Color

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
colors = {
    'cyan': "#00ffff", 'darkblue': "#00008b", #'darkcyan': "#008b8b",
    'darkmagenta': "#8b008b", 'darkolivegreen': "#556b2f", 'darkorange': "#ff8c00",
    'darkgrey': "#a9a9a9", 'darkgreen': "#006400", 'darkkhaki': "#bdb76b",
    'darkorchid': "#9932cc", 'darkred': "#8b0000", 'darksalmon': "#e9967a",
    'black': "#000000", 'blue': "#0000ff", 'brown': "#a52a2a",
    'aqua': "#00ffff", 'azure': "#f0ffff", 'beige': "#f5f5dc",
    'darkviolet': "#9400d3", 'fuchsia': "#ff00ff", 'gold': "#ffd700",
    'green': "#008000", 'indigo': "#4b0082", 'khaki': "#f0e68c",
    # 'lightblue': "#add8e6", 'lightcyan': "#e0ffff", 'lightgreen': "#90ee90",
    # 'lightgrey': "#d3d3d3", 'lightpink': "#ffb6c1", 'lightyellow': "#ffffe0",
    'lime': "#00ff00", 'magenta': "#ff00ff", 'maroon': "#800000",
    'navy': "#000080", 'olive': "#808000", 'orange': "#ffa500",
    'pink': "#ffc0cb", 'purple': "#800080", 'violet': "#800080",
    'red': "#ff0000", #'silver': "#c0c0c0", #'white': "#ffffff",
    #'yellow': "#ffff00"
}
pallet += list(colors.values())

# Station_markSize = 30
Station_markSize = 20
LocPD_dotSize = 5
SHOW_LABEL = False
SHOW_FLOW = True
SHOW_BUNDLE_POLY = True
SHOW_PD = True
SHOW_PD_ARROW = False
SHOW_HT_XHT = True
MRT_MARK = False



def drawLabel(qp, label, cx, cy, w, h):
    if SHOW_LABEL:
        qp.translate(cx - w / 2, cy - h / 2)
        label.drawContents(qp, QRectF(0, 0, w, h))
        qp.translate(-(cx - w / 2), -(cy - h / 2))


class Singapore(object):
    def __init__(self, sgBoarderXY, sgDistrictXY):
        self.sgBoarderXY = sgBoarderXY
        self.sgDistrictXY = sgDistrictXY
        self.sgDistrictPolyXY = {}
        for dist_name, points in self.sgDistrictXY.items():
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
        pen = QPen(Qt.black, 0.2, Qt.DashLine)
        qp.setPen(pen)
        for dist_name, poly in self.sgDistrictXY.items():
            self.drawPoly(qp, poly)
        pen = QPen(Qt.black, 1)
        qp.setPen(pen)
        for _, poly in enumerate(self.sgBoarderXY):
            self.drawPoly(qp, poly)


class Network(object):
    thickness = 3.0
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

    def __init__(self, mrtLines):
        self.lines = []
        for lineName, points in mrtLines:
            self.lines.append([lineName, [QPoint(x, y) for x, y in points]])

    def draw(self, qp):
        for lineName, (p0, p1) in self.lines:
            qp.setPen(Network.linePen[lineName])
            qp.drawLine(p0, p1)


class Station(object):
    font = QFont('Decorative', 15, italic=True)
    labelH = 30
    unit_labelW = 20

    def __init__(self, sid, STN, cx, cy):
        self.label = QTextDocument()
        self.label.setHtml("S<sub>%d</sub>" % sid)
        self.label.setDefaultFont(Station.font)
        self.labelW = len("Sx") * Station.unit_labelW
        #
        self.STN = STN
        self.cPoint = QPoint(cx - Station_markSize / 2, cy - Station_markSize / 2)

        self.pixmap = QPixmap('mrtMark.png').scaledToWidth(Station_markSize)

    def draw(self, qp):
        if MRT_MARK:
            qp.drawPixmap(self.cPoint, self.pixmap)
        else:

            # pen_color = QColor("#d3d3d3")
            pen_color = QColor(Color('grey').get_hex_l())


            pen = QPen(pen_color, 2, Qt.SolidLine)
            qp.setPen(pen)
            qp.setBrush(pen_color)
            qp.drawRect(self.cPoint.x(),
                        self.cPoint.y(),
                        Station_markSize, Station_markSize)


        drawLabel(qp, self.label,
                  self.cPoint.x(),
                  self.cPoint.y(), self.labelW, Station.labelH)


class Flow(object):
    lineProp = 50

    def __init__(self, weight, points):
        self.weight = weight
        self.points = [QPoint(x, y) for x, y in points]

    def draw(self, qp):
        if SHOW_FLOW:
            pen = QPen(Qt.black, self.weight * Flow.lineProp, Qt.SolidLine)
            qp.setPen(pen)
            for i in range(len(self.points) - 1):
                p0 = self.points[i]
                p1 = self.points[i + 1]
                qp.drawLine(p0, p1)


class Task(object):
    arrow_HS, arrow_VS = 15, 10
    font = QFont('Decorative', 12)
    unit_labelW = 20
    labelH = 30

    def __init__(self, tid, pXY, dXY, randomPair=False):
        self.tid = tid
        self.pcx, self.pcy = pXY
        self.dcx, self.dcy = dXY
        self.includedInBundle = False
        self.randomPair = randomPair
        if not self.randomPair:
            self.labelW = (len("%dx" % tid)) * Task.unit_labelW
            self.label_info = []
            pLabel = QTextDocument()
            pLabel.setHtml("%d<sup>+</sup>" % tid)
            pLabel.setDefaultFont(Task.font)
            self.label_info.append([pLabel, self.pcx, self.pcy])
            dLabel = QTextDocument()
            dLabel.setHtml("%d<sup>-</sup>" % tid)
            dLabel.setDefaultFont(Task.font)
            self.label_info.append([dLabel, self.pcx, self.pcy])
            #
            x0, y0, x1, y1 = self.pcx, self.pcy, self.dcx, self.dcy
            self.arrowLine = [[x0, y0, x1, y1]]
            ax, ay = x1 - x0, y1 - y0
            la = np.sqrt(ax ** 2 + ay ** 2)
            ux, uy = ax / la, ay / la
            px, py = -uy, ux
            self.arrowLine.append([x1, y1,
                                   x1 - (ux * Task.arrow_HS) + (px * Task.arrow_VS),
                                   y1 - (uy * Task.arrow_HS) + (py * Task.arrow_VS)])
            self.arrowLine.append([x1, y1,
                                   x1 - (ux * Task.arrow_HS) - (px * Task.arrow_VS),
                                   y1 - (uy * Task.arrow_HS) - (py * Task.arrow_VS)])

    def draw(self, qp):
        return None

        if not self.randomPair:
            for label, x, y in self.label_info:
                drawLabel(qp, label,
                          x + LocPD_dotSize / 2, y,
                          self.labelW, Task.labelH)
            #
            pen_color = QColor(pallet[self.tid % len(pallet)])
            pen = QPen(pen_color, 2, Qt.SolidLine)
            qp.setPen(pen)
            if SHOW_HT_XHT and self.includedInBundle:
                qp.setBrush(Qt.NoBrush)
                # qp.setBrush(pen_color)
            else:
                qp.setBrush(Qt.NoBrush)
            # if SHOW_PD:
            #     pen = QPen(Qt.black, 2, Qt.SolidLine)
            #     qp.setPen(pen)
            #     for x0, y0, x1, y1 in [
            #         (self.pcx, self.pcy, self.pcx + LocPD_dotSize, self.pcy + LocPD_dotSize),
            #         (self.pcx + LocPD_dotSize, self.pcy, self.pcx, self.pcy + LocPD_dotSize),
            #         (self.dcx, self.dcy, self.dcx + LocPD_dotSize, self.dcy + LocPD_dotSize),
            #         (self.dcx + LocPD_dotSize, self.dcy, self.dcx, self.dcy + LocPD_dotSize)
            #     ]:
            #         qp.drawLine(x0, y0, x1, y1)

            # if SHOW_PD and not self.includedInBundle:
            if SHOW_PD and self.includedInBundle:



                qp.drawEllipse(self.pcx - LocPD_dotSize / 2, self.pcy - LocPD_dotSize / 2,
                               LocPD_dotSize, LocPD_dotSize)
                qp.drawRect(self.dcx - LocPD_dotSize / 2, self.dcy - LocPD_dotSize / 2,
                            LocPD_dotSize, LocPD_dotSize)
            # if SHOW_PD_ARROW:
            # if SHOW_PD_ARROW and not self.includedInBundle:
            if SHOW_PD_ARROW and self.includedInBundle:
                for x0, y0, x1, y1 in self.arrowLine:
                    qp.drawLine(x0, y0, x1, y1)
        # else:
        pen = QPen(Qt.black, 2, Qt.SolidLine)
        qp.setPen(pen)
        for x0, y0, x1, y1 in [
                                (self.pcx, self.pcy, self.pcx + LocPD_dotSize, self.pcy + LocPD_dotSize),
                                (self.pcx + LocPD_dotSize, self.pcy, self.pcx, self.pcy + LocPD_dotSize),
                                (self.dcx, self.dcy, self.dcx + LocPD_dotSize, self.dcy + LocPD_dotSize),
                                (self.dcx + LocPD_dotSize, self.dcy, self.dcx, self.dcy + LocPD_dotSize)
                            ]:
            qp.drawLine(x0, y0, x1, y1)


class Bundle(object):
    font = QFont('Decorative', 15)
    labelH = 30
    unit_labelW = 15
    polyLineTH = 4
    maxFP_TH = 4

    def __init__(self, bid, bc, bndlPoly, edgeFeasiblity=None):
        self.bid = bid
        #
        self.label = QTextDocument()
        _assTaskIDs = str(bc)
        self.label.setHtml("b<sub>%d</sub>: %s" % (self.bid, _assTaskIDs))
        self.label.setDefaultFont(Bundle.font)
        self.labelW = (len("bx: ") + len(_assTaskIDs)) * Bundle.unit_labelW
        #
        self.bnldPoly = bndlPoly
        c = np.array(list(map(sum, zip(*bndlPoly)))) / len(bndlPoly)
        self.lx, self.ly = c[0], c[1]
        self.edgeFeasiblity = {}
        if edgeFeasiblity:
            maxCount = np.array(list(edgeFeasiblity.values())).max()
            for k, numCount in edgeFeasiblity.items():
                self.edgeFeasiblity[k] = numCount / float(maxCount) * Bundle.maxFP_TH

    def draw(self, qp):
        drawLabel(qp, self.label,
                  self.lx, self.ly, self.labelW, Bundle.labelH)
        if SHOW_BUNDLE_POLY:
            if self.bid != 3:
                return None
            pen = QPen(QColor(pallet[self.bid % len(pallet)]), Bundle.polyLineTH, Qt.DashDotLine)
            qp.setPen(pen)
            for i in range(len(self.bnldPoly) - 1):
                x0, y0 = self.bnldPoly[i]
                x1, y1 = self.bnldPoly[i + 1]
                qp.drawLine(x0, y0, x1, y1)
            x0, y0 = self.bnldPoly[len(self.bnldPoly) - 1]
            x1, y1 = self.bnldPoly[0]
            qp.drawLine(x0, y0, x1, y1)
        #
        for (x0, y0, x1, y1), thickness in self.edgeFeasiblity.items():
            pen = QPen(Qt.black, thickness)
            qp.setPen(pen)
            qp.drawLine(x0, y0, x1, y1)
