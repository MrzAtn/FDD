import cv2
import numpy as np
import module.functions
import module.object_Tracker
import module.line_Tracker

class EventClassifier:

    buffer_position = {}
    hist_event = {}
    
    acc = {
        "start_frame" : None,
        "end_frame" : None,
        "temp_acc" : 0,
        "acc_length" : 0, # compteur en seconde
        "acc_wait_threshold" : 30, # compteur en frame
        "acc_min_length" : 20, # compteur en seconde
    }

    noAcc = {
        "temp_noAcc" : 0,
        "noAcc_length" : 0, # compteur en seconde
        "noAcc_wait_threshold" : 5, # compteur en frame
        "noAcc_min_length" : 20, # compteur en seconde
    }

    crossing = {
        "start_frame" : None,
        "end_frame" : None,
        "temp_crossing" : 0,
        "crossing_duration" : 40,# compteur en frame
        "crossing_threshold" : 5
    }

    CICO = {
        "cut_duration" : 40,
        "start_frame" : None,
        "end_frame" : None,
        "temp_CICO" : 0,
        "cut_threshold" : 5
    }

    Cu_or_Ci = {
        "cut_duration" : 20,
        "temp_Cu_or_Ci" : 0,
        "start_frame" : None,
        "end_frame" : None,
        "cut_threshold" : 5
    }

    def __init__(self, object_Tracker, line_Tracker):
       self.object_Tracker = object_Tracker
       self.line_Tracker = line_Tracker

    def register_data(self, videoName, frame_count):
        self.videoName = videoName
        self.frame_count = frame_count
        for i, obj in self.object_Tracker.objects.items():
            d_L = obj[4]-obj[2]
            d_l = obj[3]-obj[1]
            ratio = d_L / d_l
            # Relevé de la voie de chaque objet
            if i in self.object_Tracker.leftAdjacentLane:
                lane = "left"
            elif i in self.object_Tracker.rightAdjacentLane:
                lane = "right"
            elif i in self.object_Tracker.egoLane:
                lane = "ego"
            else: 
                break
            # self.buffer_position[i].append([ratio, lane])
            if i in self.buffer_position:
                # ajout des données
                self.buffer_position[i].insert(0, [ratio, lane])
                if len(self.buffer_position[i]) > 40:
                    del self.buffer_position[i][-1] # on supprime la plus ancienne data
            else:
                # création d'un buffer pour l'objet
                self.buffer_position[i] = [[ratio, lane]]

    def d_Event_ACC(self):
        if len(self.object_Tracker.egoLane) > 0:
            self.acc["object_ID"] = self.object_Tracker.egoLane[0]
            self.acc["temp_acc"] = self.acc["acc_wait_threshold"]
        # Si aucun objet n'est détecté comme ACC à la frame actuelle, on lance le compteur pour le noAcc 
        if self.acc["temp_acc"] != self.acc["acc_wait_threshold"]:
            self.noAcc["temp_noAcc"] = self.noAcc["noAcc_wait_threshold"]
        if self.acc["temp_acc"] > 0:
            self.acc["temp_acc"] -= 1
            self.acc["acc_length"] += 1
        if self.noAcc["temp_noAcc"] > 0:
            self.noAcc["temp_noAcc"] -= 1
            self.noAcc["noAcc_length"] += 1
        if self.noAcc["noAcc_length"] == self.noAcc["noAcc_min_length"]: 
            event_type = "ACC"
            start_frame = self.frame_count - self.noAcc["noAcc_length"]
            end_frame = self.frame_count
            try:
                self.hist_event[self.videoName][event_type]["start_frame"] = np.append(self.hist_event[self.videoName][event_type]["start_frame"], start_frame)
                self.hist_event[self.videoName][event_type]["end_frame"] = np.append(self.hist_event[self.videoName][event_type]["start_frame"], end_frame)
            except:
                self.hist_event[self.videoName] = {}
                self.hist_event[self.videoName][event_type] = {}
                self.hist_event[self.videoName][event_type]["start_frame"] = np.array([start_frame])
                self.hist_event[self.videoName][event_type]["end_frame"] = np.array([end_frame])

    def d_Event_Crossing(self):
        # Analyse des voies sur 5 frames pour chaque objet
        for id_object, data in self.buffer_position.items():
            if len(data) == self.crossing["crossing_duration"]:
                sum_ratio = 0
                count_laneChange = 0
                for index, sub_data in enumerate(data):
                    sum_ratio += sub_data[0]
                    try:
                        if data[index][1] != data[index+1][1]: # L'objet change de voie
                            count_laneChange += 1
                    except: # Pour le dernier élément du tableau de data on ne le compare pas
                        pass
                average_sum = sum_ratio/len(data)
                # Register data from event detection
                if count_laneChange == 2 and average_sum > 1.5:
                    i = 0
                    self.crossing["temp_crossing"] = 0
                    while (data[i][1] == data[i+1][1]):
                        self.crossing["temp_crossing"] += 1
                        i += 1
                    if self.crossing["temp_crossing"] >= self.crossing["crossing_threshold"]:
                        event_type = "CROSSING_"+ str(data[-1][1]) # Left or Right
                        start_frame = self.frame_count - self.crossing["crossing_duration"]
                        end_frame = self.frame_count
                        try:
                            self.hist_event[self.videoName][event_type]["start_frame"] = np.append(self.hist_event[self.videoName][event_type]["start_frame"], start_frame)
                            self.hist_event[self.videoName][event_type]["end_frame"] = np.append(self.hist_event[self.videoName][event_type]["start_frame"], end_frame)
                        except:
                            self.hist_event[self.videoName] = {}
                            self.hist_event[self.videoName][event_type] = {}
                            self.hist_event[self.videoName][event_type]["start_frame"] = np.array([start_frame])
                            self.hist_event[self.videoName][event_type]["end_frame"] = np.array([end_frame])
                        # Deletion of the object that has just performed the event to launch a new follow-up
                        self.buffer_position[id_object] = []

    def d_Event_CI_or_CO(self):
        # Analyse des voies sur 5 frames pour chaque objet
        for id_object, data in self.buffer_position.items():
            if len(data) > self.Cu_or_Ci["cut_duration"]:
                sum_ratio = 0
                count_laneChange = 0
                for index, sub_data in enumerate(data):
                    sum_ratio += sub_data[0]
                    try:
                        if data[index][1] != data[index+1][1]: # L'objet change de voie
                            count_laneChange += 1
                    except: # Pour le dernier élément du tableau de data on ne le compare pas
                        pass
                if count_laneChange == 1:
                    i = 0
                    self.Cu_or_Ci["temp_Cu_or_Ci"] = 0
                    while (data[i][1] == data[i+1][1]):
                        self.Cu_or_Ci["temp_Cu_or_Ci"] += 1
                        i += 1
                    if self.Cu_or_Ci["temp_Cu_or_Ci"] >= self.Cu_or_Ci["cut_threshold"]:
                        # Case of cut in situation
                        if data[0][1] == "ego":
                            event_type = "CUT_INT"
                        # Case of cut out situation
                        else:
                            event_type = "CUT_OUT"
                        start_frame = self.frame_count - self.Cu_or_Ci["cut_duration"]
                        end_frame = self.frame_count
                        try:
                            self.hist_event[self.videoName][event_type]["start_frame"] = np.append(self.hist_event[self.videoName][event_type]["start_frame"], start_frame)
                            self.hist_event[self.videoName][event_type]["end_frame"] = np.append(self.hist_event[self.videoName][event_type]["end_frame"], end_frame)
                        except:
                            self.hist_event[self.videoName] = {}
                            self.hist_event[self.videoName][event_type] = {}
                            self.hist_event[self.videoName][event_type]["start_frame"] = np.array([start_frame])
                            self.hist_event[self.videoName][event_type]["end_frame"] = np.array([end_frame])
                        # Deletion of the object that has just performed the event to launch a new follow-up
                        self.buffer_position[id_object] = []

    def d_Event_CICO(self):
        # Analyse des voies sur 5 frames pour chaque objet
        for id_object, data in self.buffer_position.items():
            if len(data) > self.CICO["cut_duration"]:
                sum_ratio = 0
                count_laneChange = 0
                for index, sub_data in enumerate(data):
                    sum_ratio += sub_data[0]
                    try:
                        if data[index][1] != data[index+1][1]: # L'objet change de voie
                            count_laneChange += 1
                    except: # Pour le dernier élément du tableau de data on ne le compare pas
                        pass
                average_sum = sum_ratio/len(data)
                # Register data from event detection
                if count_laneChange == 2 and average_sum < 1.5:
                    i = 0
                    self.CICO["temp_CICO"] = 0
                    while (data[i][1] == data[i+1][1]):
                        self.CICO["temp_CICO"] += 1
                        i += 1
                    if self.CICO["temp_CICO"] >= self.CICO["crossing_threshold"]:
                        event_type = "CICO"
                        start_frame = self.frame_count - self.CICO["cut_duration"]
                        end_frame = self.frame_count
                        try:
                            self.hist_event[self.videoName][event_type]["start_frame"] = np.append(self.hist_event[self.videoName][event_type]["start_frame"], start_frame)
                            self.hist_event[self.videoName][event_type]["end_frame"] = np.append(self.hist_event[self.videoName][event_type]["start_frame"], end_frame)
                        except:
                            self.hist_event[self.videoName] = {}
                            self.hist_event[self.videoName][event_type] = {}
                            self.hist_event[self.videoName][event_type]["start_frame"] = np.array([start_frame])
                            self.hist_event[self.videoName][event_type]["end_frame"] = np.array([end_frame])
                        # Deletion of the object that has just performed the event to launch a new follow-up
                        self.buffer_position[id_object] = []

    def update(self, videoName, frame_count, flag_acc, flag_cico, flag_crossing, flag_cut):
        self.register_data(videoName, frame_count)
        if flag_acc == True:
            self.d_Event_ACC()
        if flag_crossing == True:
            self.d_Event_Crossing()
        if flag_cico == True:
            self.d_Event_CICO()
        if flag_cut == True:
            self.d_Event_CI_or_CO()