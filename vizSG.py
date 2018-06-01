import os.path as opath
import os
import sys
import pickle
from functools import reduce
import numpy as np
#
from PyQt5.QtWidgets import QWidget, QApplication, QShortcut
from PyQt5.QtGui import (QPainter, QPen, QColor, QFont, QTextDocument,
                         QImage, QPainterPath, QKeySequence, QPixmap, QPalette, QCursor)
from PyQt5.QtCore import Qt, QSize, QPoint, QRectF, QSizeF
from PyQt5.QtPrintSupport import QPrinter
from colour import Color
from shapely.geometry import Polygon, Point
#
from __path_organizer import exp_dpath
from sgDistrict import get_sgBorder, get_distPoly
from sgLocationPD import pdLoc, get_locationPD
from sgMRT import get_coordMRT, get_mrtNet, get_mrtNetNX, get_route

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

from random import seed

seed(1)

colors = {
    'cyan': "#00ffff", 'darkblue': "#00008b",'darkcyan': "#008b8b",
    'darkmagenta': "#8b008b", 'darkolivegreen': "#556b2f", 'darkorange': "#ff8c00",
    'darkgrey': "#a9a9a9", 'darkgreen': "#006400", 'darkkhaki': "#bdb76b",
    'darkorchid': "#9932cc", 'darkred': "#8b0000", 'darksalmon': "#e9967a",
    'black': "#000000", 'blue': "#0000ff", 'brown': "#a52a2a",
    'aqua': "#00ffff", 'azure': "#f0ffff", 'beige': "#f5f5dc",
    'darkviolet': "#9400d3", 'fuchsia': "#ff00ff", 'gold': "#ffd700",
    'green': "#008000", 'indigo': "#4b0082", 'khaki': "#f0e68c",
    'lightblue': "#add8e6", 'lightcyan': "#e0ffff", 'lightgreen': "#90ee90",
    'lightgrey': "#d3d3d3", 'lightpink': "#ffb6c1", 'lightyellow': "#ffffe0",
    'lime': "#00ff00", 'magenta': "#ff00ff", 'maroon': "#800000",
    'navy': "#000080", 'olive': "#808000", 'orange': "#ffa500",
    'pink': "#ffc0cb", 'purple': "#800080", 'violet': "#800080",
    'red': "#ff0000", 'silver': "#c0c0c0", 'white': "#ffffff",
    'yellow': "#ffff00"
}


pallet += list(colors.values())






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

SHOW_ALL_PD = True
SHOW_MRT_LINE = False
SHOW_DISTRICT = True
SHOW_LABEL = False
SAVE_IMAGE = True

# LocPD_dotSize = 10
# Station_markSize = 25

LocPD_dotSize = 20
Station_markSize = 20


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


def drawLabel(qp, label, cx, cy, w, h):
    if SHOW_LABEL:
        qp.translate(cx - w / 2, cy - h / 2)
        label.drawContents(qp, QRectF(0, 0, w, h))
        qp.translate(-(cx - w / 2), -(cy - h / 2))


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
            pen = QPen(Qt.black, 0.2, Qt.DashLine)
            qp.setPen(pen)
            for dist_name, poly in self.sgDistrictXY.items():
                self.drawPoly(qp, poly)
        pen = QPen(Qt.black, 1)
        qp.setPen(pen)
        for _, poly in enumerate(self.sgBoarderXY):
            self.drawPoly(qp, poly)


class Station(object):
    font = QFont('Decorative', 15, italic=True)
    labelH = 30
    unit_labelW = 20

    def __init__(self, sid, STN, lat, lng):
        self.label = QTextDocument()
        self.label.setHtml("S<sub>%d</sub>" % sid)
        self.label.setDefaultFont(Station.font)
        self.labelW = len("Sx") * Station.unit_labelW
        #
        self.STN = STN
        cx, cy = convert_GPS2xy(lng, lat)
        self.cPoint = QPoint(cx - Station_markSize / 2, cy - Station_markSize / 2)

        self.pixmap = QPixmap('mrtMark.png').scaledToWidth(Station_markSize)

    def draw(self, qp):
        qp.drawPixmap(self.cPoint, self.pixmap)

        # drawLabel(qp, self.label,
        #           self.cPoint.x(),
        #           self.cPoint.y(), self.labelW, Station.labelH)


class Pair(object):
    arrow_HS, arrow_VS = 10, 5

    def __init__(self, tid, pLoc, dLoc):
        self.tid = tid
        _, lat, lng, _, _, _, _ = pLoc
        self.pcx, self.pcy = convert_GPS2xy(lng, lat)
        #
        _, lat, lng, _, _, _, _ = dLoc
        self.dcx, self.dcy = convert_GPS2xy(lng, lat)

        x0, y0, x1, y1 = self.pcx, self.pcy, self.dcx, self.dcy
        self.lines = [[x0, y0, x1, y1]]
        ax, ay = x1 - x0, y1 - y0
        la = np.sqrt(ax ** 2 + ay ** 2)
        ux, uy = ax / la, ay / la
        px, py = -uy, ux
        self.lines.append([x1, y1,
                           x1 - (ux * Pair.arrow_HS) + (px * Pair.arrow_VS),
                           y1 - (uy * Pair.arrow_HS) + (py * Pair.arrow_VS)])
        self.lines.append([x1, y1,
                           x1 - (ux * Pair.arrow_HS) - (px * Pair.arrow_VS),
                           y1 - (uy * Pair.arrow_HS) - (py * Pair.arrow_VS)])


    def draw(self, qp):
        # if (self.tid % 20) % 2 == 0:
        pen = QPen(QColor(pallet[self.tid]), 2, Qt.SolidLine)

        # else:
        #     pen = QPen(QColor(pallet[self.tid]), 2, Qt.DotLine)
        qp.setPen(pen)
        qp.setBrush(Qt.NoBrush)
        qp.drawEllipse(self.pcx - LocPD_dotSize / 2, self.pcy - LocPD_dotSize / 2,
                         LocPD_dotSize, LocPD_dotSize)

        qp.drawRect(self.dcx - LocPD_dotSize / 2, self.dcy - LocPD_dotSize / 2,
                       LocPD_dotSize, LocPD_dotSize)



        for x0, y0, x1, y1 in self.lines:
            qp.drawLine(x0, y0, x1, y1)









class LocPD(object):
    font = QFont('Decorative', 12)
    labelH = 30
    unit_labelW = 20
    #
    pen = QPen(Qt.black, 1, Qt.SolidLine)

    def __init__(self, tid, typePD, lat, lng, nearestMRT, Duration, Location, District):
        self.tid = tid
        self.label = QTextDocument()
        if typePD == 'P':
            self.label.setHtml("%d<sup>+</sup>" % tid)
        else:
            self.label.setHtml("%d<sup>-</sup>" % tid)
        self.label.setDefaultFont(LocPD.font)
        self.labelW = len("0x") * LocPD.unit_labelW
        #
        self.typePD = typePD
        self.cx, self.cy = convert_GPS2xy(lng, lat)

    def draw(self, qp):
        # qp.setPen(LocPD.pen)

        if (self.tid % 40) % 2 == 0:
            pen = QPen(QColor(pallet[self.tid % len(pallet)]), 2, Qt.SolidLine)

        else:
            pen = QPen(QColor(pallet[self.tid % len(pallet)]), 2, Qt.DotLine)
        qp.setPen(pen)
        qp.setBrush(Qt.NoBrush)
        self.drawingFunc = qp.drawEllipse if self.typePD == 'P' else qp.drawRect
        self.drawingFunc(self.cx - LocPD_dotSize / 2, self.cy - LocPD_dotSize / 2,
              LocPD_dotSize, LocPD_dotSize)

        drawLabel(qp, self.label,
                  self.cx + LocPD_dotSize / 2, self.cy,
                  self.labelW, LocPD.labelH)


class Flow(object):
    lineProp = 40

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
    polyLineTH = 4

    def __init__(self, bid, assTaskIDs, points):
        self.bid = bid
        #
        self.label = QTextDocument()
        _assTaskIDs = str(assTaskIDs)
        self.label.setHtml("b<sub>%d</sub>: %s" % (self.bid, _assTaskIDs))
        self.label.setDefaultFont(Bundle.font)
        self.labelW = (len("bx: ") + len(_assTaskIDs)) * Bundle.unit_labelW
        #
        self.points = points
        c = np.array(list(map(sum, zip(*points)))) / len(points)
        self.lx, self.ly = c[0], c[1]

    def draw(self, qp):


        # drawLabel(qp, self.label,
        #           self.lx, self.ly, self.labelW, Bundle.labelH)



        pen = QPen(QColor(pallet[self.bid % len(pallet)]), Bundle.polyLineTH, Qt.DashDotDotLine)
        qp.setPen(pen)
        for i in range(len(self.points) - 1):
            x0, y0 = self.points[i]
            x1, y1 = self.points[i + 1]
            qp.drawLine(x0, y0, x1, y1)
        x0, y0 = self.points[len(self.points) - 1]
        x1, y1 = self.points[0]
        qp.drawLine(x0, y0, x1, y1)


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
            from random import shuffle, sample
            shuffle(locationPD)
            numTasks = 0

            while numTasks < 15:
                ploc, dloc = sample(locationPD, 2)
                self.objForDrawing.append(Pair(numTasks, ploc, dloc))
                numTasks += 1




            # for i, o in enumerate(locationPD):
            #     if i > 40:
            #         continue
            #
            #     _, lat, lng, nearMRT, duration, loc, district = o
            #     type_PD = 'P' if i < 20 else 'D'
            #     # self.objForDrawing.append(LocPD(i % 30, *o))
            #     self.objForDrawing.append(LocPD(i % 40,
            #                 type_PD, lat, lng, nearMRT, duration, loc, district))
        if SHOW_MRT_LINE:
            for i, (STN, (lat, lng)) in enumerate(mrt_coords.items()):
                self.objForDrawing.append(Station(i, STN, lat, lng))
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
            mrts = set()
            for k, (mrt0, mrt1) in enumerate(flow_oridest):
                route = get_route(mrtNetNX, mrt0, mrt1)
                self.objForDrawing.append(Flow(w_k[k], route, mrt_coords))
                #
                if mrt0 not in mrts:
                    lat0, lng0 = mrt_coords[mrt0]
                    self.objForDrawing.append(Station(len(mrts), mrt0, lat0, lng0))
                    mrts.add(mrt0)
                if mrt1 not in mrts:
                    lat1, lng1 = mrt_coords[mrt1]
                    self.objForDrawing.append(Station(len(mrts), mrt1, lat1, lng1))
                    mrts.add(mrt1)
            #
            ln_locO = {o.Location: o for o in locationPD}
            task_pdO = []
            for tid, (pLoc, dLoc) in enumerate(task_ppdp):
                _, lat, lng, nearMRT, duration, loc, district = ln_locO[pLoc]
                po = LocPD(tid, 'P', lat, lng, nearMRT, duration, loc, district)
                _, lat, lng, nearMRT, duration, loc, district = ln_locO[dLoc]
                do = LocPD(tid, 'D', lat, lng, nearMRT, duration, loc, district)


                # self.objForDrawing.append(po)
                # self.objForDrawing.append(do)


                task_pdO.append([po, do])
            #
            if 'CWL' in pkl_files['sol']:
                C, q_c = [drawingInfo['sol'].get(k) for k in ['C', 'q_c']]
                generatedBundles = [C[c] for c in range(len(C)) if q_c[c] > 0.5]
            else:
                assert 'GH' in pkl_files['sol']
                cB_M = drawingInfo['prmts']['cB_M']
                bc = drawingInfo['sol']['bc']
                generatedBundles = [o for o in bc if cB_M <= len(o)]
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
            self.image.fill(Qt.white)  ## switch it to else
            pal = self.palette()
            pal.setColor(QPalette.Background, Qt.white)
            self.setAutoFillBackground(True)
            self.setPalette(pal)
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
        printer = QPrinter(QPrinter.HighResolution)
        # printer.setPageSize(QPrinter.A4)
        printer.setPageSizeMM(QSizeF(WIDTH / 15, HEIGHT / 15))
        printer.setFullPage(True)
        printer.setPageMargins(0.0, 0.0, 0.0, 0.0, QPrinter.Millimeter)
        printer.setColorMode(QPrinter.Color)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(img_fpath[:-len('.png')] + '.pdf')
        pixmap = self.grab().scaledToHeight(
            printer.pageRect(QPrinter.DevicePixel).size().toSize().height() * 2)
        painter = QPainter(printer)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()


    def drawCanvas(self, qp):
        for o in self.objForDrawing:
            o.draw(qp)
        if self.mousePressed:
            qp.translate(self.px, self.py)
            self.selDistLabel.drawContents(qp, QRectF(0, 0, self.labelW, Viz.labelH))
            qp.translate(-self.px, -self.py)


def testSingle():
    dplym_dpath = reduce(opath.join, [exp_dpath, '_summary', 'dplym'])
    prmts_dpath = reduce(opath.join, [exp_dpath, '_summary', 'prmts'])
    sol_dpath = reduce(opath.join, [exp_dpath, '_summary', 'sol'])
    #
    prefix = 'mrtS1_dt80'
    aprc = 'CWL1'
    # pkl_files = {
    #     'dplym': opath.join(dplym_dpath, 'dplym_%s.pkl' % prefix),
    #     'prmts': opath.join(prmts_dpath, 'prmts_%s.pkl' % prefix),
    #     'sol': opath.join(sol_dpath, 'sol_%s_%s.pkl' % (prefix, aprc))
    # }

    pkl_files = {}

    app = QApplication(sys.argv)
    viz = Viz(pkl_files)
    if SAVE_IMAGE:
        # viz.save_img('%s_%s.png' % (prefix, aprc))
        viz.save_img('SG.png')
    sys.exit(app.exec_())


def gen_imgs():
    dplym_dpath = reduce(opath.join, [exp_dpath, '_summary', 'dplym'])
    prmts_dpath = reduce(opath.join, [exp_dpath, '_summary', 'prmts'])
    sol_dpath = reduce(opath.join, [exp_dpath, '_summary', 'sol'])
    viz_dpath = reduce(opath.join, [exp_dpath, '_summary', 'viz'])
    if not opath.exists(viz_dpath):
        os.mkdir(viz_dpath)
    #
    aprcs = ['GH'] + ['CWL%d' % cwl_no for cwl_no in range(5, 0, -1)]
    for fn in os.listdir(prmts_dpath):
        if fn == 'prmts_mrtS1_dt80.pkl':
            continue
        if not fn.endswith('.pkl'): continue
        _, prefix = fn[:-len('.pkl')].split('_')
        #
        dplym_fpath = opath.join(dplym_dpath, 'dplym_%s.pkl' % prefix)
        prmts_fpath = opath.join(prmts_dpath, fn)
        for i, aprc in enumerate(aprcs):
            print(aprc, prefix)
            sol_fpath = opath.join(sol_dpath, 'sol_%s_%s.pkl' % (prefix, aprc))
            viz_fpath = opath.join(viz_dpath, '%s_%s.png' % (prefix, aprc))
            if opath.exists(viz_fpath):
                continue
            if not opath.exists(sol_fpath):
                continue
            pkl_files = {
                'dplym': dplym_fpath,
                'prmts': prmts_fpath,
                'sol': sol_fpath
            }
            app = QApplication(sys.argv)
            viz = Viz(pkl_files)
            viz.save_img(viz_fpath)
            app.quit()
            del app


if __name__ == '__main__':
    testSingle()
    # gen_imgs()

