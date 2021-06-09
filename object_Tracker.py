## Import dependent packages ##
from scipy.spatial import distance as dist
from operator import itemgetter, attrgetter
from collections import OrderedDict
import numpy as np
from module.functions import*

class Object_Tracker:

    nextObjectID = 0 # Variable dedicated to fixe a single ID for each object detection
    objects = OrderedDict() # Dict to store deteted object => {"ID": [Values]}
    objectsSorted = list()
    disappeared = OrderedDict() # Dict to store disapear object
    egoLane, leftAdjacentLane, rightAdjacentLane = list(), list(), list() # Create list in order to store object per lane 

    def __init__(self, maxDisappearedEgo=10, maxDisappearedSide=2):
        self.maxDisappearedEgo = maxDisappearedEgo # Store the number of max consecutive frame where the object is not visible
        self.maxDisappearedSide = maxDisappearedSide # Store the number of max consecutive frame where the object is not visible

    # Method to store a new object 
    def register(self, inputObject):
        self.objects[self.nextObjectID] = inputObject
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    # Method to unsave specific object 
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
    
    def laneAssignment(self, lt_lines):
         # eset for each step
        self.objectsSorted = list()
        self.egoLane, self.leftAdjacentLane, self.rightAdjacentLane = list(), list(), list()
        # On tri dans un premier temps par rapport aux bas des boxes
        for ids, data in self.objects.items():
            self.objectsSorted.append([ids, data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]])
        self.objectsSorted = sorted(self.objectsSorted, key=lambda colonne:colonne[4])

        for object_detected in self.objectsSorted:
                if lt_lines["left"] is not None and lt_lines["right"] is not None: 
                    if get_x_onfit(lt_lines["left"][0], object_detected[3]) < average(object_detected[2], object_detected[4]) < get_x_onfit(lt_lines["right"][0], object_detected[3]):
                        self.egoLane.append(object_detected[0])
                    elif  get_x_onfit(lt_lines["left"][0], object_detected[3]) > average(object_detected[2], object_detected[4]):
                        self.leftAdjacentLane.append(object_detected[0])
                    elif average(object_detected[2], object_detected[4]) > get_x_onfit(lt_lines["right"][0], object_detected[3]):
                        self.rightAdjacentLane.append(object_detected[0])
                elif lt_lines["left"] is not None:
                    if get_x_onfit(lt_lines["left"][0], object_detected[3]) < object_detected[2] < get_x_onfit(lt_lines["left"][0], object_detected[3])+40:
                        self.egoLane.append(object_detected[0])
                    elif get_x_onfit(lt_lines["left"][0], object_detected[3]) > object_detected[4]:
                        self.leftAdjacentLane.append(object_detected[0])
                    elif get_x_onfit(lt_lines["left"][0], object_detected[3])+40 < object_detected[2]:
                        self.rightAdjacentLane.append(object_detected[0])
                elif lt_lines["right"] is not None:
                    if get_x_onfit(lt_lines["right"][0], object_detected[3])-40 < object_detected[4] < get_x_onfit(lt_lines["right"][0], object_detected[3]):
                        self.egoLane.append(object_detected[0])
                    elif get_x_onfit(lt_lines["right"][0], object_detected[3])-40 < object_detected[4]:
                        self.leftAdjacentLane.append(object_detected[0])
                    elif  object_detected[2] > get_x_onfit(lt_lines["right"][0], object_detected[3]):
                        self.rightAdjacentLane.append(object_detected[0])

    def update(self, rects, lt_lines): # rects == list of tuples (?, xmin, ymin, xmax, ymax, pourcentage, classe)
        self.laneAssignment(lt_lines)
        if len(rects) == 0: # We check if there are no detection, in this case we increment the dict
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if objectID in self.egoLane: 
                    # If dont see the specific object over the max variable we have to delete it 
                    if self.disappeared[objectID] > self.maxDisappearedEgo:
                        self.deregister(objectID)
                if objectID in self.leftAdjacentLane or objectID in self.rightAdjacentLane :
                    if self.disappeared[objectID] > self.maxDisappearedSide:
                        self.deregister(objectID)
            return self.objects
        # initialize an array of input centroids for the current frame
        inputObjects = np.zeros((len(rects), len(rects[0])+2)) # form: [(ID, (cX, cY))]
        inputCentroids = np.zeros((len(rects), 2), dtype="int") # form: [(ID, (cX, cY))]
        # loop over the bounding box rectangles
        for (i, (_, xmin, ymin, xmax, ymax, pourcentage, classe)) in enumerate(rects):
            # use the bounding box coordinates to compute the centroid
            cX = int((xmin + xmax) / 2.0)
            cY = int((ymin + ymax) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputObjects[i] = (_, xmin, ymin, xmax, ymax, pourcentage, classe, cX, cY)
        if len(self.objects) == 0: # if we havent any object detected before 
            for i in range(0, len(inputObjects)):
                self.register(inputObjects[i])
        else: # uptdate current detected object 
            # We get the prev object id and data 
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # Compute the distance between old and new centroid
            D = dist.cdist(np.array(objectCentroids)[:, -2:], inputCentroids)
            rows = D.min(axis=1).argsort() # return an array with the ID of the colum in croissant order 
            cols = D.argmin(axis=1)[rows] # return the coord(col, row) of the lowest matrix value
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    # Si les centroids sont deja traité alors on les pass
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputObjects[col]
                self.disappeared[objectID] = 0
                # Indicate examined 
                usedRows.add(row)
                usedCols.add(col)
            # Indicate unexamined 
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]: # row (index 0) are old object and cols (index 1) are new input
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if objectID in self.egoLane: 
                        # If dont see the specific object over the max variable we have to delete it 
                        if self.disappeared[objectID] > self.maxDisappearedEgo:
                            self.deregister(objectID)
                    elif objectID in (self.leftAdjacentLane or self.rightAdjacentLane) :
                        if self.disappeared[objectID] > self.maxDisappearedSide:
                            self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputObjects[col])
        return self.objects