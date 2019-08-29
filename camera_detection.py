from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtGui
import sysw
import cv2
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QApplication


class MainApp(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        self.video_size = QSize(640, 480)
        self.setup_ui()

    def setup_ui(self):
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.video_size)

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)

        self.start_button = QPushButton("Start Normal Cam")
        self.start_button.clicked.connect(self.setup_camera)


        self.test_button = QPushButton("Start Color Detection Cam")
        self.test_button.clicked.connect(self.color_detect)



        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.start_button)
        self.main_layout.addWidget(self.test_button)
        self.main_layout.addWidget(self.quit_button)

        self.setLayout(self.main_layout)

    def setup_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size.height())

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(100)




    def display_video_stream(self):
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        image = QImage(frame, frame.shape[1], frame.shape[0],
                       frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image))




    def color_detect(self):
        lower = {'red': (166, 84, 141), 'green': (66, 122, 129), 'blue': (97, 100, 117), 'yellow': (23, 59, 119),
                 'orange': (0, 50, 80),'dark_green':(5,115,0),'black':(1,1,1),'white':(180,180,180)}  # assign new item lower['blue'] = (93, 10, 0)

        upper = {'red': (186, 255, 255), 'green': (86, 255, 255), 'blue': (117, 255, 255),'yellow': (54, 255, 255),
                 'orange': (20, 255, 255),'dark_green':(12,230,0),'black':(70,70,70),'white':(255,255,255)}

        colors = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0), 'yellow': (0, 255, 217),
                  'orange': (0, 140, 255),'dark_green':(5,115,0),'black':(0,0,0),'white':(255,255,255)}


        camera = cv2.VideoCapture(0)

        while True:
            (grabbed, frame) = camera.read()
            frame = cv2.flip(frame, 1)

            blurred = cv2.GaussianBlur(frame, (1, 1), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            for key, value in upper.items():
                mask = cv2.inRange(hsv, lower[key], upper[key])

                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)[-2]

                if len(cnts) > 0:
                    c = max(cnts, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)

                    if radius > 0.5 :
                        cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                        cv2.putText(frame, key + "", (int(x - radius), int(y - radius)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, colors[key])

            cv2.imshow("Color Detection ",frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()





if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.setWindowTitle("Color Detection System AI Application")
    win.setWindowIcon(QtGui.QIcon("0.png"))
    win.show()
    sys.exit(app.exec_())
