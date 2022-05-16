import cv2, os
import numpy as np
from utility import process_point

class ImageCroper():
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.height, self.width, _ = self.image.shape

    def expand(self, polygon, w_rate = 0, h_rate = 1/100):        
        #padding
        rect = cv2.boundingRect(polygon)
        _,__,pw,ph = rect

        expand_pw = w_rate * pw
        expand_ph = h_rate * ph
        expand = np.float32([expand_pw, expand_ph])
        
        #get expand four corners
        top_left = np.round(polygon.min(axis = 0) - expand, 0)
        bottom_right = np.round(polygon.max(axis = 0) + expand, 0)
        
        xmin, ymin = process_point(top_left, self.height, self.width)
        xmax, ymax = process_point(bottom_right, self.height, self.width)

        return xmin, ymin, xmax, ymax

    #Crop image under rectangel shape
    def crop_rectangle(self, polygon):
        '''
            polygon: numpy arrays (dtype: int32, shape: (n,2), n: so luong point)
            image: pixel arrays
        '''
        polygon = np.float32(polygon)
        xmin, ymin, xmax, ymax = self.expand(polygon)
        croped_image = self.image[ymin : ymax+1, xmin : xmax+1].copy()
        
        return croped_image


    #Crop image under polygon shape
    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        pts = np.asarray(pts, dtype = "float32")
        (tl, tr, br, bl) = pts

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def crop_polygon(self, polygon):
        polygon = np.float32(polygon)
        ## (1) Crop the bounding rect
        rect = cv2.boundingRect(polygon)
        x,y,w,h = rect

        ##Crop image
        image = cv2.imread(self.image_path)
        croped_image = image[y:y+h, x:x+w].copy()

        ## (2) make mask
        ##Adjust points
        polygon = np.int32(polygon)
        pts = polygon - polygon.min(axis=0)

        mask = np.zeros(croped_image.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        ## (3) do bit-op
        dst = cv2.bitwise_and(croped_image, croped_image, mask=mask)
        wraped = self.four_point_transform(dst, pts)

        return wraped
