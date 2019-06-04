import cv2 as cv
import threading


class Camera(object):
    def __init__(self):
        self._cam = None
        self._frame = None
        self._frame_width = None
        self._frame_height = None
        self._ret = False

        self._auto_undist = False
        self._cam_matrix = None
        self._dist_coeff = None

        self._is_running = False

    def _init_cam(self):
        pass

    def start_cam(self):
        self._init_cam()
        self._is_running = True
        threading.Thread(target=self._update_cam, args=()).start()

    def _read_from_cam(self):
        if self._cam is None:
            raise Exception('Camera isn\'t initialized')

    def _update_cam(self):
        while True:
            if self._is_running:
                self._ret, self._frame = self._read_from_cam()
            else:
                break

    def get_w_h(self):
        return self._frame_width, self._frame_height

    def read(self):
        if self._is_running:
            return self._ret, self._frame
        else:
            import warnings
            warnings.warn('Start camera with start_cam()')
            return False, None

    def release(self):
        self._is_running = False

    def is_running(self):
        return self._is_running

    def set_calibration_matrices(self, camera_matrix, distortion_coeff):
        self._cam_matrix = camera_matrix
        self._dist_coeff = distortion_coeff

    def activate_auto_undistortion(self):
        self._auto_undist = True

    def deactivate_auto_undistortion(self):
        self._auto_undist = False

    def _undistort_image(self, image):
        if self._cam_matrix is None or self._dist_coeff is None:
            import warnings
            warnings.warn("Undistortion has no effect because <camera_matrix>/<distortion_coefficients> is None!")
            return image

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(self._cam_matrix,
                                                              self._dist_coeff, (w, h),
                                                              1,
                                                              (w, h))
        undistorted = cv.undistort(image, self._cam_matrix, self._dist_coeff, None,
                                   new_camera_matrix)
        return undistorted

    def __enter__(self):
        self.start_cam()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
