import sys
import cv2
import rospy
import cv_bridge
from sensor_msgs.msg import Image
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

class ImageProcessor(QObject):
    image_signal1 = pyqtSignal(QImage)
    image_signal2 = pyqtSignal(QImage)

    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.bridge = cv_bridge.CvBridge()
        self.subscriber1 = rospy.Subscriber('/fr3/camera/image_raw', Image, self.callback1, queue_size=1)
        self.subscriber2 = rospy.Subscriber('/fr3/camera2/image_raw', Image, self.callback2, queue_size=1)

    def callback1(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        qt_image = self.convert_image(cv_image)
        self.image_signal1.emit(qt_image)

    def callback2(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        qt_image = self.convert_image(cv_image)
        self.image_signal2.emit(qt_image)

    def convert_image(self, cv_image):
        height, width, channel = cv_image.shape
        bytesPerLine = 3 * width
        return QImage(cv_image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

class MainWindow(QLabel):
    def __init__(self, title="Camera View"):
        super(MainWindow, self).__init__()
        self.setWindowTitle(title)
        self.resize(640, 480)

    @pyqtSlot(QImage)
    def set_image(self, image):
        pixmap = QPixmap.fromImage(image)
        self.setPixmap(pixmap)

if __name__ == "__main__":
    rospy.init_node("camera_viewer", anonymous=True)
    app = QApplication(sys.argv)
    
    # Camera 1 Window
    window1 = MainWindow("Top Camera View")
    # Camera 2 Window
    window2 = MainWindow("Front Camera View")

    processor = ImageProcessor()
    processor.image_signal1.connect(window1.set_image)
    processor.image_signal2.connect(window2.set_image)

    window1.show()
    window2.show()

    sys.exit(app.exec_())
