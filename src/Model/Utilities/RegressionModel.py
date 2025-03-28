from abc import abstractmethod
from utilities import ToMetric
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Regression():
    def __init__(self, outerDiameter, mapDim=1000, threshold=0.0, diskFilterRadius=20):
        self.outerDiameter = outerDiameter
        self.mapDim = mapDim
        self.gridSize = 1.5
        self.pixelDiameter = int(self.mapDim / self.gridSize)
        self.mapX, self.mapY = np.meshgrid(np.linspace(-self.gridSize, self.gridSize, mapDim), np.linspace(-self.gridSize, self.gridSize, mapDim))
        self.threshold = threshold
        self.diskFilterRadius = diskFilterRadius
        self.rdot = self.calcR()

    @abstractmethod
    def generate_grain_geometry(self):
        """Generates a grain geometry based on the given parameters"""

    def normalize(self, value):
        """Transforms real unit quantities into self.mapX, self.mapY coordinates. For use in indexing into the
        coremap."""
        return value / (0.5 * self.outerDiameter)

    def unNormalize(self, value):
        """Transforms self.mapX, self.mapY coordinates to real unit quantities. Used to determine real lengths in coremap."""
        return (value / 2) * self.outerDiameter

    def mapToLength(self, value):
        """Converts pixels to meters. Used to extract real distances from pixel distances such as contour lengths"""
        return self.outerDiameter * (value / self.pixelDiameter)

    def mapToArea(self, value):
        """Used to convert sq pixels to sqm. For extracting real areas from the regression map."""
        return (self.outerDiameter ** 2) * (value / ((self.pixelDiameter) ** 2))

    def findContour(self, img):
        """Finds countours in the given image"""
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def drawAllContours(self, base_img, all_contours):
        """Draws all the contours with different colors"""
        # convert to color
        if len(base_img.shape) == 3 and base_img.shape[2] == 3:
            colored_img = base_img.copy()
        else:
            colored_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

        # list of colors to use for contours
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        # iterate through contours and draw them using color list
        for i, contours in enumerate(all_contours):
            cv2.drawContours(colored_img, contours, -1, colors[i % len(colors)], 1)
        return colored_img

    def applyDiskFilter(self, img):
        """Applies a disk filter to the image"""
        radius = self.diskFilterRadius + 1
        # generate disk kernel for filter
        y, x = np.ogrid[-radius: radius+1, -radius: radius+1]
        mask = x**2 + y**2 <= radius**2
        kernel = np.zeros((2*radius+1, 2*radius+1))
        kernel[mask] = 1
        kernel /= kernel.sum()  # normalize
        
        # apply filter
        filtered_img = cv2.filter2D(img, -1, kernel)
        return filtered_img

    def applyThreshold(self, img):
        """Applies a experiementally determined threshold to the blurred image to create a binary image"""
        if self.threshold < 0 or self.threshold > 1:
            raise ValueError("Threshold value must be between 0 and 1")
        _, thresholded_img = cv2.threshold(img, self.threshold * 255, 255, cv2.THRESH_BINARY)
        return thresholded_img

    def contourIntersectsBoundary(self, contour, center, radius):
        """Determines if a contour intersects with a circular boundary"""
        for point in contour:
            distance = np.linalg.norm(point[0] - center)
            if distance > radius:
                return True
        return False

    def getCorePerimeter(self, contours):
        """Used to calculate the initial core perimeter in pixels """
        # Calculates the perimeter of the countour
        firstPass = cv2.arcLength(contours[0], True)
        # Douglas-Peucker Algorithm for Shape Approximation to 0.1 % error
        epsilon = 0.001*firstPass
        DPfit = cv2.approxPolyDP(contours[0],epsilon,True)

        # Recalculates the perimeter of the countour
        perimeter = cv2.arcLength(DPfit, True)

        return perimeter

    def getCoreArea(self, contours):
        """Used to calculate the Initial core area, A_port, in pixels"""
        area = cv2.contourArea(contours[0])
        return area
    
    def calcR(self):
        """Approximates the 'rate' of regression by applying the regression process to a simple image and determining the change in white pixels"""
        def countWhitePixels(img):
            # counts white pixels in a single row
            return np.sum(img[1, :] == 255)
        # creates a simple image that is half black pixels on one side and white on the other
        half_black_white = np.hstack((np.zeros((self.mapDim, self.mapDim//2), dtype=np.uint8), 255*np.ones((self.mapDim, self.mapDim//2), dtype=np.uint8)))
        # counts white pixels before applying regression
        white_pixel_count1 = countWhitePixels(half_black_white)
        # applies regression process to the image
        blurred_img = self.applyDiskFilter(half_black_white)
        thresholded_img = self.applyThreshold(blurred_img)
        # counts white pixels after applying regression
        white_pixel_count2 = countWhitePixels(thresholded_img)
        # returns the difference in white pixels as the rate of regression
        return white_pixel_count2 - white_pixel_count1

    def showImage(self, img):
        """Displays the image"""
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def runLoop(self, rStop=-1):
        """Loop through the regression process"""
        # generate initial grain geometry before applying regression
        base_img = self.generate_grain_geometry()

        # initial setup for loop
        all_contours = []
        contours = self.findContour(base_img)
        all_contours.append(contours)
        processed_img = base_img.copy()
        outer_diameter = self.pixelDiameter
        outer_radius = int(outer_diameter/2)
        center = (processed_img.shape[0] // 2, processed_img.shape[1] // 2)

        data_columns = [
            "r", "pixels", "area", "perimeter"
        ]
        df = pd.DataFrame(columns=data_columns)
        step = 0
        r = 0
        while True:
            # Apply filter and threshold to simulate regression
            disk_filtered_img = self.applyDiskFilter(processed_img)
            thresholded_img = self.applyThreshold(disk_filtered_img)

            # Check if contour intersects the fuel grain boundary and stop looping if it does
            contours = self.findContour(thresholded_img)
            if not contours or any(self.contourIntersectsBoundary(cont, center, outer_radius) for cont in contours):
                break

            # Used to stop regression if the contours start exceeding the boundary
            mask = np.zeros_like(thresholded_img)
            cv2.circle(mask, center, outer_radius, (255), -1)
            masked_thresholded_img = cv2.bitwise_and(thresholded_img, thresholded_img, mask=mask)

            # find contour and save it to list
            contour = self.findContour(masked_thresholded_img)
            all_contours.append(contour)

            # calculate values for burnback table and store it in a df
            perimeter = self.getCorePerimeter(contour)
            area = self.getCoreArea(contour)
            r = r + self.rdot
            new_data = {
                "r": self.mapToLength(r),
                "pixels": r,
                "area": self.mapToArea(area),
                "perimeter": self.mapToLength(perimeter)
            }
            df.loc[len(df)] = new_data

            # prepare for next iteration
            processed_img = masked_thresholded_img.copy()

            # img = masked_thresholded_img.copy()
            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # cv2.circle(img, center, outer_radius, (0, 0, 255), 3)
            # cv2.drawContours(img, contour, -1, (0, 255, 0), 3)
            # if rStop > 0 and self.mapToLength(r) >= rStop:
            #     break

        df.to_csv("src/Model/CSV/burnback_table.csv", index=False)

        colored_img = self.drawAllContours(base_img, all_contours)
        cv2.circle(colored_img, center, outer_diameter // 2, (0, 0, 255), 3)
        self.showImage(colored_img)
        return df
    
class StarGeometry(Regression):
    def __init__(self, outer_diameter, num_points, point_length, point_base_width, offset=(0,0), mapDim=1000, threshold=0.36, diskFilterRadius=20):
        super().__init__(outer_diameter, mapDim, threshold, diskFilterRadius)
        self.outer_diameter = outer_diameter
        self.num_points = num_points
        self.point_length = point_length
        self.point_base_width = point_base_width
        self.offset = offset

    def generate_grain_geometry(self):
        num_points = self.num_points
        point_length = self.normalize(self.point_length)
        point_base_width = self.normalize(self.point_base_width)
        offsetX = self.normalize(self.offset[0])
        offsetY = self.normalize(self.offset[1])

        # Create a white image of the desired dimensions
        img = np.ones((mapDim, mapDim), dtype=np.uint8) * 0
        
        mapX = self.mapX - offsetX
        mapY = self.mapY + offsetY
        
        for i in range(0, num_points):
            theta = 2 * np.pi / num_points * i
            comp0 = np.cos(theta)
            comp1 = np.sin(theta)

            rect = abs(comp0 * mapX + comp1 * mapY)
            width = point_base_width / 2 * (1 - (((mapX ** 2 + mapY ** 2) ** 0.5) / point_length))
            vect = rect < width
            near = comp1 * mapX - comp0 * mapY > -0.025
            img[np.logical_and(vect, near)] = 255  # Set to white
        return img
    
class FinocylGeometry(Regression):
    def __init__(self, outer_diameter, inner_diameter, num_fins, fin_length, fin_width, mapDim=1000, threshold=0.36, diskFilterRadius=20):
        super().__init__(outer_diameter, mapDim, threshold, diskFilterRadius)
        self.outer_diameter = outer_diameter
        self.inner_diameter = inner_diameter
        self.num_fins = num_fins
        self.fin_length = fin_length
        self.fin_width = fin_width

    def generate_grain_geometry(self):
        innerDiameter = self.normalize(self.inner_diameter)
        finLength = self.normalize(self.fin_length)
        finWidth = self.normalize(self.fin_width)
        numFins = self.num_fins
        mapDim = self.mapDim

        img = np.ones((mapDim, mapDim), dtype=np.uint8) * 0

        # Open up core
        img[self.mapX**2 + self.mapY**2 < (innerDiameter / 2)**2] = 255

        # Add fins
        for i in range(0, numFins):
            theta = 2 * np.pi / numFins * i
            vect0 = np.cos(theta)
            vect1 = np.sin(theta)
            vect = abs(vect0*self.mapX + vect1*self.mapY) < finWidth / 2
            near = (vect1 * self.mapX) - (vect0 * self.mapY) > 0
            far = (vect1 * self.mapX) - (vect0 * self.mapY) < finLength
            ends = np.logical_and(far, near)
            img[np.logical_and(vect, ends)] = 255
        return img

if __name__ == "__main__":
    outer_diameter = ToMetric(3.375, 'in')
    inner_diameter = ToMetric(1.0, 'in')
    fin_length = ToMetric(0.9+0.5, 'in')
    fin_width = ToMetric(0.2, 'in')
    num_fins = 6

    mapDim = 2500
    finocyl = FinocylGeometry(outer_diameter=outer_diameter, inner_diameter=inner_diameter, num_fins=num_fins, fin_length=fin_length, fin_width=fin_width, mapDim=mapDim, threshold=0.36, diskFilterRadius=40)
    df = finocyl.runLoop(rStop=-1)

    plt.figure()
    plt.plot(df['r'], df['area']*1000)
    plt.xlabel('Radius (m)')
    plt.ylabel('Area (m^2)')
    plt.title('Area vs Radius')

    plt.figure()
    plt.plot(df['r'], df['perimeter']*1000)
    plt.xlabel('Radius (m)')
    plt.ylabel('Perimeter (m)')
    plt.title('Perimeter vs Radius')

    plt.show()