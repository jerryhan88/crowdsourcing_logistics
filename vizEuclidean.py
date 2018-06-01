import os.path as opath
import sys
import numpy as np
import pickle

from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import (QPen, QColor, QFont, QTextDocument, QPainter, QImage, QPainterPath)
from PyQt5.QtCore import Qt, QSize, QRectF

D360 = 360.0

pallet = [
        "#0000ff",  # blue
        "#a52a2a",  # brown
        "#ff00ff",  # magenta
        "#008000",  # green
        "#4b0082",  # indigo
        "#f0e68c",  # khaki
        "#800000",  # maroon
        "#000080",  # navy
        "#ffa500",  # orange
        "#ffc0cb",  # pink
        "#ff0000",  # red
        "#808080",  # grey
]


mainFrameOrigin = (100, 150)
frameSize = (800, 800)

WITH_SOL = True


def load_pklFiles(dplym_fpath, prmts_fpath, sol_fpath):
    container = []
    for fpath in [dplym_fpath, prmts_fpath, sol_fpath]:
        if opath.exists(fpath):
            with open(fpath, 'rb') as fp:
                container.append(pickle.load(fp))
        else:
            container.append(None)
    return {k: container[i] for i, k in enumerate(['dplym', 'prmts', 'sol'])}


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


def drawLabel(qp, label, cx, cy, w, h):
    qp.translate(cx - w / 2, cy - h / 2)
    label.drawContents(qp, QRectF(0, 0, w, h))
    qp.translate(-(cx - w / 2), -(cy - h / 2))


class Path(object):
    arrow_HS, arrow_VS = 10, 5
    font = QFont('Decorative', 12, italic=True)
    labelW, labelH = 25, 25
    pen = QPen(Qt.black, 1, Qt.SolidLine)

    def __init__(self, pid, baseInputs, drawingInputs):
        self.pid = pid
        #
        self.label = QTextDocument()
        self.label.setHtml("k<sub>%d</sub>" % self.pid)
        self.label.setDefaultFont(Path.font)
        #
        self.oriX, self.oriY, destX, destY = drawingInputs
        self.lines = [[self.oriX, self.oriY, destX, destY]]
        ax, ay = destX - self.oriX, destY - self.oriY
        la = np.sqrt(ax ** 2 + ay ** 2)
        ux, uy = ax / la, ay / la
        px, py = -uy, ux
        self.lines.append([destX, destY,
                           destX - (ux * Path.arrow_HS) + (px * Path.arrow_VS),
                           destY - (uy * Path.arrow_HS) + (py * Path.arrow_VS)])
        self.lines.append([destX, destY,
                           destX - (ux * Path.arrow_HS) - (px * Path.arrow_VS),
                           destY - (uy * Path.arrow_HS) - (py * Path.arrow_VS)])

    def draw(self, qp):
        # qp.setFont(Path.font)
        drawLabel(qp, self.label,
                  self.oriX, self.oriY, Path.labelW, Path.labelH)
        #
        qp.setPen(Path.pen)
        qp.setBrush(Qt.NoBrush)
        for x0, y0, x1, y1 in self.lines:
            qp.drawLine(x0, y0, x1, y1)


class Task(object):
    font = QFont('Decorative', 10)
    labelW, labelH = 20, 20
    pen = QPen(Qt.black, 0.5, Qt.DashLine)
    dotSize = 18

    def __init__(self, tid, baseInputs, drawingInputs):
        self.tid = tid
        #
        self.plabel, self.dlabel = QTextDocument(), QTextDocument()
        self.plabel.setHtml("%d<sup>+</sup>" % self.tid)
        self.dlabel.setHtml("%d<sup>-</sup>" % self.tid)
        self.plabel.setDefaultFont(Task.font)
        self.dlabel.setDefaultFont(Task.font)
        self.ppX, self.ppY, self.dpX, self.dpY = drawingInputs

    def draw(self, qp):
        # qp.setFont(Task.font)
        for cx, cy, label in [(self.ppX, self.ppY, self.plabel),
                              (self.dpX, self.dpY, self.dlabel)]:
            drawLabel(qp, label,
                      cx, cy, Task.labelW, Task.labelH)

        qp.setPen(Task.pen)
        qp.setBrush(Qt.NoBrush)
        for f, (p0, p1) in [(qp.drawEllipse, (self.ppX, self.ppY)),
                            (qp.drawRect, (self.dpX, self.dpY))]:
            f(p0 - Task.dotSize / 2, p1 - Task.dotSize / 2,
                Task.dotSize, Task.dotSize)


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
        drawLabel(qp, self.label,
                  self.lx, self.ly, self.labelW, Bundle.labelH)
        pen = QPen(QColor(pallet[self.bid % len(pallet)]), 0.5, Qt.DashLine)
        qp.setPen(pen)
        for i in range(len(self.points) - 1):
            x0, y0 = self.points[i]
            x1, y1 = self.points[i + 1]
            qp.drawLine(x0, y0, x1, y1)
        x0, y0 = self.points[len(self.points) - 1]
        x1, y1 = self.points[0]
        qp.drawLine(x0, y0, x1, y1)


class Viz(QWidget):
    def __init__(self, viz_input, img_fpath='temp.png'):
        super().__init__()
        self.img_fpath = img_fpath
        path_oridest, task_ppdp = viz_input['dplym']
        B, T = [viz_input['prmts'].get(k) for k in ['B', 'T']]
        z_bi = viz_input['sol']['z_bi']
        #
        self.paths = []
        for i, (ori, dest) in enumerate(path_oridest):
            oriX, oriY = ori[0] * frameSize[0], ori[1] * frameSize[1]
            destX, destY = dest[0] * frameSize[0], dest[1] * frameSize[1]
            drawingInputs = oriX, oriY, destX, destY
            self.paths.append(Path(i, None, drawingInputs))
        self.tasks, b_in_t = [], {}
        for i, (pp, dp) in enumerate(task_ppdp):
            ppX, ppY = pp[0] * frameSize[0], pp[1] * frameSize[1]
            dpX, dpY = dp[0] * frameSize[0], dp[1] * frameSize[1]
            drawingInputs = ppX, ppY, dpX, dpY
            self.tasks.append(Task(i, None, drawingInputs))
            for b in B:
                if z_bi[b, i] > 0.5:
                    if b not in b_in_t:
                        b_in_t[b] = []
                    b_in_t[b].append([i, drawingInputs])
        #
        if WITH_SOL:
            self.bundles = []
            for b, taskInfo in b_in_t.items():
                assTaskIDs, points = [], []
                for i, drawingInputs in taskInfo:
                    assTaskIDs.append(i)
                    ppX, ppY, dpX, dpY = drawingInputs
                    points.append([ppX, ppY])
                    points.append([dpX, dpY])
                points = sort_clockwise(points)
                self.bundles.append(Bundle(b, assTaskIDs, points))
        #
        w, h = frameSize
        self.image = QImage(w, h, QImage.Format_RGB32)
        self.path = QPainterPath()
        self.image.fill(Qt.white)  ## switch it to else
        self.update()
        #
        self.initUI()

    def initUI(self):
        w, h = frameSize
        self.setGeometry(mainFrameOrigin[0], mainFrameOrigin[1], w, h)
        self.setWindowTitle('Viz')
        self.setFixedSize(QSize(w, h))
        self.show()

    def save_img(self):
        self.image.save(self.img_fpath, 'png')

    def paintEvent(self, e):
        for dev in [self, self.image]:
            qp = QPainter()
            qp.begin(dev)
            self.drawCanvas(qp)
            qp.end()
        # qp = QPainter()
        # qp.begin(self)
        # self.drawCanvas(qp)
        # qp.end()

    def drawCanvas(self, qp):
        for o in self.paths + self.tasks + self.bundles:
            o.draw(qp)
        if WITH_SOL:
            for o in self.bundles:
                o.draw(qp)


if __name__ == '__main__':
    from __path_organizer import exp_dpath
    dplym_fpath = opath.join('_temp', 'dplym_euclideanDistEx0.pkl')
    prmts_fpath = opath.join('_temp', 'prmts_euclideanDistEx0.pkl')
    sol_fpath = opath.join('_temp', 'sol_euclideanDistEx0_EX1.pkl')
    #
    img_fpath = opath.join('_temp', 'euclideanDistEx0_EX1.png')

    # dplym_fpath = opath.join(exp_dpath, 'tempProb/dplym_nt0006-nb0002-np005-dt0.80-vt3.pkl')
    # prmts_fpath = opath.join('_temp', 'prmts_nt0006-nb0002-np005.pkl')
    # sol_fpath = opath.join('_temp', 'sol_nt0006-nb0002-np005.pkl')

    viz_input = load_pklFiles(dplym_fpath, prmts_fpath, sol_fpath)

    app = QApplication(sys.argv)
    viz = Viz(viz_input, img_fpath)
    viz.save_img()
    sys.exit(app.exec_())
