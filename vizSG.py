import os.path as opath
import os
import sys
import pickle
from functools import reduce
import numpy as np
#
from PyQt5.QtWidgets import QWidget, QApplication, QShortcut
from PyQt5.QtGui import (QPainter, QFont, QTextDocument,
                         QImage, QKeySequence, QPalette,
                         QCursor)
from PyQt5.QtCore import Qt, QSize, QRectF, QSizeF
from PyQt5.QtPrintSupport import QPrinter
#
from __path_organizer import exp_dpath
from sgDistrict import get_sgBorder, get_distPoly
from sgLocationPD import pdLoc, get_locationPD
from sgMRT import get_coordMRT, get_mrtNet, get_mrtNetNX, get_route
#
from vizSG_cls import (Singapore,
                       Network, Station,
                       Flow, Task,
                       Bundle)


from random import seed

seed(1)



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

mrt_coords = get_coordMRT()
locationPD = get_locationPD()
mrtNetNX = get_mrtNetNX()

WIDTH = 1800.0
HEIGHT = lat_gap * (WIDTH / lng_gap)

FRAME_ORIGIN = (60, 100)

SHOW_ALL_PD = False
SHOW_MRT_LINE = False
SHOW_LABEL = False
SAVE_IMAGE = True


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


class Viz(QWidget):
    font = QFont('Decorative', 15)
    labelH = 30
    unit_labelW = 15

    def __init__(self, pkl_files):
        super().__init__()
        self.app_name = 'Viz'
        self.objForDrawing = []
        #
        self.init_bgDrawing()
        if pkl_files:
            self.drawingInfo = {}
            for k, fpath in pkl_files.items():
                with open(fpath, 'rb') as fp:
                    self.drawingInfo[k] = pickle.load(fp)
            self.app_name += '-%s' % self.drawingInfo['prmt']['problemName']
            self.init_prmtDrawing()
            self.init_solDrawing(pkl_files)
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

    def init_bgDrawing(self):
        sgBoarderXY = []
        for poly in sgBorder:
            sgBorderPartial_xy = []
            for lat, lng in poly:
                x, y = convert_GPS2xy(lng, lat)
                sgBorderPartial_xy += [(x, y)]
            sgBoarderXY.append(sgBorderPartial_xy)
        sgDistrictXY = {}
        distPoly = get_distPoly()
        for dist_name, poly in distPoly.items():
            points = []
            for lat, lng in poly:
                points.append(convert_GPS2xy(lng, lat))
            sgDistrictXY[dist_name] = points
        self.sg = Singapore(sgBoarderXY, sgDistrictXY)
        self.objForDrawing = [self.sg]
        #
        if SHOW_MRT_LINE:
            mrtNet = get_mrtNet()
            mrtLines = []
            for lineName, connections in mrtNet.items():
                for paris in connections:
                    points = []
                    for mrt in paris:
                        lat, lng = mrt_coords[mrt]
                        x, y = convert_GPS2xy(lng, lat)
                        points.append([x, y])
                    mrtLines.append([lineName, points])
            self.objForDrawing.append(Network(mrtLines))
            for i, (STN, (lat, lng)) in enumerate(mrt_coords.items()):
                cx, cy = convert_GPS2xy(lng, lat)
                self.objForDrawing.append(Station(i, STN, cx, cy))
        if SHOW_ALL_PD:
            from random import shuffle, sample
            shuffle(locationPD)
            numTasks = 0
            while numTasks < 15:
                pLoc, dLoc = sample(locationPD, 2)
                _, lat, lng, _, _, _, _ = pLoc
                pcx, pcy = convert_GPS2xy(lng, lat)
                _, lat, lng, _, _, _, _ = dLoc
                dcx, dcy = convert_GPS2xy(lng, lat)
                self.objForDrawing.append(Task(numTasks,
                                               [pcx, pcy], [dcx, dcy]))
                numTasks += 1

    def init_prmtDrawing(self):
        self.flow_oridest, task_ppdp = self.drawingInfo['dplym']
        w_k = self.drawingInfo['prmt']['w_k']
        mrts = set()
        for k, (mrt0, mrt1) in enumerate(self.flow_oridest):
            route = get_route(mrtNetNX, mrt0, mrt1)
            points = []
            for mrt in route:
                lat, lng = mrt_coords[mrt]
                x, y = convert_GPS2xy(lng, lat)
                points.append([x, y])
            self.objForDrawing.append(Flow(w_k[k], points))
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
        self.task_pdO = []
        for tid, (pLoc, dLoc) in enumerate(task_ppdp):
            _, lat, lng, _, _, _, _ = ln_locO[pLoc]
            pcx, pcy = convert_GPS2xy(lng, lat)
            _, lat, lng, _, _, _, _ = ln_locO[dLoc]
            dcx, dcy = convert_GPS2xy(lng, lat)
            self.objForDrawing.append(Task(tid,
                                           [pcx, pcy], [dcx, dcy]))
            self.task_pdO.append([[pcx, pcy], [dcx, dcy]])

    def init_solDrawing(self, pkl_files):
        if not 'scFP' in pkl_files:
            if 'CWL' in pkl_files['sol']:
                C, q_c = [self.drawingInfo['sol'].get(k) for k in ['C', 'q_c']]
                selBundles = [C[c] for c in range(len(C)) if q_c[c] > 0.5]
            else:
                assert 'GH' in pkl_files['sol']
                cB_M = self.drawingInfo['prmt']['cB_M']
                bc = self.drawingInfo['sol']['bc']
                selBundles = [o for o in bc if cB_M <= len(o)]
            #
            for bid, bc in enumerate(selBundles):
                bndlPoly = []
                for tid in bc:
                    (pcx, pcy), (dcx, dcy) = self.task_pdO[tid]
                    bndlPoly.append([pcx, pcy])
                    bndlPoly.append([dcx, dcy])
                bndlPoly = sort_clockwise(bndlPoly)
                self.objForDrawing.append(Bundle(bid, bc, bndlPoly))
        else:
            scFP_fpath = pkl_files['scFP']
            bid = int(opath.basename(scFP_fpath)[len('bid'):-len('.pkl')])
            with open(pkl_files['scFP'], 'rb') as fp:
                _, bc, feasiblePath = pickle.load(fp)
            bndlPoly = []
            for tid in bc:
                (pcx, pcy), (dcx, dcy) = self.task_pdO[tid]
                bndlPoly.append([pcx, pcy])
                bndlPoly.append([dcx, dcy])
            bndlPoly = sort_clockwise(bndlPoly)
            edgeFeasiblity = {}
            for k in feasiblePath:
                mrt0, mrt1 = self.flow_oridest[k]
                route = get_route(mrtNetNX, mrt0, mrt1)
                for i in range(len(route) - 1):
                    lat, lng = mrt_coords[route[i]]
                    x0, y0 = convert_GPS2xy(lng, lat)
                    lat, lng = mrt_coords[route[i + 1]]
                    x1, y1 = convert_GPS2xy(lng, lat)
                    k = (x0, y0, x1, y1)
                    if k not in edgeFeasiblity:
                        edgeFeasiblity[k] = 0
                    edgeFeasiblity[k] += 1
            self.objForDrawing.append(Bundle(bid, bc, bndlPoly, edgeFeasiblity))

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
        if img_fpath.endswith('.png'):
            self.image.save(img_fpath, 'png')
        else:
            assert img_fpath.endswith('.pdf')
            printer = QPrinter(QPrinter.HighResolution)
            printer.setPageSizeMM(QSizeF(WIDTH / 15, HEIGHT / 15))
            printer.setFullPage(True)
            printer.setPageMargins(0.0, 0.0, 0.0, 0.0, QPrinter.Millimeter)
            printer.setColorMode(QPrinter.Color)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(img_fpath)
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


def runSingle():
    dplym_dpath = reduce(opath.join, [exp_dpath, '_summary', 'dplym'])
    prmt_dpath = reduce(opath.join, [exp_dpath, '_summary', 'prmt'])
    sol_dpath = reduce(opath.join, [exp_dpath, '_summary', 'sol'])
    viz_dpath = reduce(opath.join, [exp_dpath, '_summary', 'viz'])
    if not opath.exists(viz_dpath):
        os.mkdir(viz_dpath)
    #
    prefix = 'mrtS1-dt80'
    aprc = 'CWL1'
    pkl_files = {
        'dplym': opath.join(dplym_dpath, 'dplym_%s.pkl' % prefix),
        'prmt': opath.join(prmt_dpath, 'prmt_%s.pkl' % prefix),
        'sol': opath.join(sol_dpath, 'sol_%s_%s.pkl' % (prefix, aprc))
    }
    viz_fpath = opath.join(viz_dpath, '%s_%s.png' % (prefix, aprc))
    #
    app = QApplication(sys.argv)
    viz = Viz(pkl_files)
    viz.save_img(viz_fpath)
    #
    # # viz.save_img('SG.png')
    # # sys.exit(app.exec_())
    #
    #
    del app
    #
    selColFP_dpath = opath.join(sol_dpath, 'selColFP')
    scFP_dpath = opath.join(selColFP_dpath, 'scFP_%s_%s' % (prefix, aprc))
    for i, fn in enumerate(os.listdir(scFP_dpath)):
        scFP_fpath = opath.join(scFP_dpath, fn)
        viz_fpath = opath.join(viz_dpath, '%s_%s_bid%d.png' % (prefix, aprc, i))
        pkl_files['scFP'] = scFP_fpath
        app = QApplication(sys.argv)
        viz = Viz(pkl_files)
        viz.save_img(viz_fpath)
        app.quit()
        del app


def gen_imgs():
    dplym_dpath = reduce(opath.join, [exp_dpath, '_summary', 'dplym'])
    prmt_dpath = reduce(opath.join, [exp_dpath, '_summary', 'prmt'])
    sol_dpath = reduce(opath.join, [exp_dpath, '_summary', 'sol'])
    viz_dpath = reduce(opath.join, [exp_dpath, '_summary', 'viz'])
    if not opath.exists(viz_dpath):
        os.mkdir(viz_dpath)
    #
    aprcs = ['GH'] + ['CWL%d' % cwl_no for cwl_no in range(5, 0, -1)]
    for fn in os.listdir(prmt_dpath):
        if fn == 'prmt_mrtS1-dt80.pkl':
            continue
        if not fn.endswith('.pkl'): continue
        _, prefix = fn[:-len('.pkl')].split('_')
        #
        dplym_fpath = opath.join(dplym_dpath, 'dplym_%s.pkl' % prefix)
        prmt_fpath = opath.join(prmt_dpath, fn)
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
                'prmt': prmt_fpath,
                'sol': sol_fpath
            }
            app = QApplication(sys.argv)
            viz = Viz(pkl_files)
            viz.save_img(viz_fpath)
            app.quit()
            del app

            selColFP_dpath = opath.join(sol_dpath, 'selColFP')
            scFP_dpath = opath.join(selColFP_dpath, 'scFP_%s_%s' % (prefix, aprc))
            for fn in os.listdir(scFP_dpath):
                viz_fpath

                scFP_fpath = opath.join(scFP_dpath, fn)
                pkl_files['scFP'] = scFP_fpath
                app = QApplication(sys.argv)
                viz = Viz(pkl_files)
                viz.save_img(viz_fpath)
                app.quit()
                del app





if __name__ == '__main__':
    runSingle()
    # gen_imgs()
