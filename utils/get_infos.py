import math

class Calcs:
    def __init__(self, horizontal_fov: float, vertical_fov: float, camera_resolution: tuple[int, int]):
        """
        Class used the make calcs

        Atributes
        ---------
        horizontal_fov : float
            The horizontal (X axis) field of vision (FOV) of the camera used to detect objects
        vertical_fov : float
            The vertical (Y axis) field of vision (FOV) of the camera used to detect objects
        camera_resolution : tuple[int, int]
            The resolution of the camera used to detect objects
        """

        self.vertical_fov = vertical_fov
        self.horizontal_fov = horizontal_fov
        self.camera_resolution = camera_resolution

    @staticmethod
    def calculate_heigth_fov(h: float, d: float, get_radians: bool = False) -> float:
        """
        Calculates the heigth (horizontal or X axis) fov (H)

        :param h: the height of the object in meters
        :param d: the distance of the camera of the object in meters
        :param get_radians: if you want to get the value in radians or degrees

        :return: the height fov in radians or degrees
        """

        h = h/2
        fov = 2 * math.atan(h/d)

        return fov if get_radians else math.degrees(fov)
    
    @staticmethod
    def calculate_width_fov(w: float, d: float, get_radians: bool = False) -> float:
        """
        Calculates the width (vertical or Y axis) fov (W)

        :param w: the width of the object in meters
        :param d: the distance of the camera of the object in meters
        :param get_radians: if you want to get the value in radians or degrees
        
        :return: the width fov in radians or degrees
        """

        w = w/2
        fov = 2 * math.atan(w/d)

        return fov if get_radians else math.degrees(fov)

    def get_horizontal_angle_offset(self, horizontal_pixel_offset: float, get_radians: bool = False):
        """
        Calculate the horizontal angle offset in degrees or radians

        :param horizontal_pixel_offset: distance of the object center of the camera resolution center

        :return: horizontal displacement angle offset
        """

        offset = (horizontal_pixel_offset / self.camera_resolution[0]) * self.horizontal_fov
        return math.radians(offset) if get_radians else offset
    
    def get_vertical_angle_offset(self, vertical_pixel_offset: float, get_radians: bool = False):
        """
        Calculate the vertical angle offset in degrees or radians

        :param vertical_pixel_offset: distance of the object center of the camera resolution center

        :return: vetical deslocation angle offset
        """

        offset = (vertical_pixel_offset / self.camera_resolution[1]) * self.vertical_fov
        return math.radians(offset) if get_radians else offset