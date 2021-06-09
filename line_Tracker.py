import numpy as np
from module.functions import*
import cv2
from math import*

class LineTracker:
	lane_frame_width = 720  # lanenet input size
	lane_frame_height = 256  # lanenet input size

	frame_width = 720  # after maskframe
	frame_height = 255  # after maskframe

	upper_limit = 175  # pixels upper limit for pixels detection
	upper_limit_default = 190
	max_dist = 10  # max distance allowed to other line's pixels to associate current pixel to the same line
	nb_close_to = 5  # min nb of pixels needed around pixel to associate current pixel to the same line
	min_height_line_points = 50  # min pixels height needed to accept line detection
	min_nb_points_start = 50  # min nb of pixels needed from same line to start finding other pixels associated to the same line
	lane_limit_up = lane_frame_height - int(lane_frame_height * (0.5 / 10))  # upper limit of starting box

	deg_poly = 2

	# Init the kalman filter object for line detection
	# k = kalman_Filter.Kalman_Filter(dim=4)

	# Setup lines default starting boxes
	param_box = {
		"left_lane_left": int(frame_width*(3/16)),
		"left_lane_right": int(frame_width*(5/16)),
		"right_lane_left": int(frame_width*(9/16)),
		"right_lane_right": int(frame_width*(11/16))
	}

	output = {
		# Left param 
		"left": None,
		"left_plot_x": None,
		"left_plot_y": None,
		"bool_left": False,
		"y_min_left": 0,
		"x_min_left": lane_frame_width,
		"x_max_left": 0,
		# Right param
		"right": None,
		"right_plot_x": None,
		"right_plot_y": None,
		"bool_right": False,
		"y_min_right": 0,
		"x_min_right": lane_frame_width,
		"x_max_right": 0,
	}

	# Buffer
	lines = {"left": None, "right": None} # Store the computed lines coord
	disappeared = {"left": None, "right": None } # Dict to store disapear line

	# Define the default coef for lines computing
	fit_default = { 
		"left": [0, -1.4, 550],
		"right": [0, 1.4, 120],
	}

	# Define extremun in oder to filtering the wrong prediction
	fit_extremum = { 
		"left": [0.5, -20, 800],
		"right": [0.5, 20, -100],
	}

	# Line_Tracker constructor
	def __init__(self, bufferSize=5):
		self.maxDisappeared = bufferSize
		self.bufferSize = bufferSize

	# Enregistre une nouvelle ligne
	def register(self, inputLine): # inputLine = [nameLine, fit_param]
		# Si la lignes n'a pas été détectée depuis au moins 5 steps ou n'a jms étée détectées.
		self.lines[inputLine[0]] = inputLine[1] # [fit_param]
		self.disappeared[inputLine[0]] = 0

	# Supprime le suivi d'une ligne
	def deregister(self, nameLine):
		self.lines[nameLine] = None
		self.disappeared[nameLine] = None

	# Permet d'actualiser le stockage des lignes
	def maj_plot_lines(self, nameLine, inputLine): # inputLine == [fit_par]
		# si on a deja 5 lignes en mémoire alors on supprime la derniere
		if len(self.lines[nameLine]) == self.bufferSize:
			del self.lines[nameLine][self.bufferSize -1]
		# On ajoute ensuite la valeur de la ligne courante dans notre list
		self.lines[nameLine].insert(0, inputLine)

	# Initialise la recherche des lines en analysant les boxes d'interet
	def initSearch(self, nameLine, binary_image):
		if np.sum(binary_image[self.lane_limit_up:,self.param_box[f"{nameLine}_lane_left"]:self.param_box[f"{nameLine}_lane_right"]]) > self.min_nb_points_start:
			self.output[nameLine][self.lane_limit_up:,self.param_box[f"{nameLine}_lane_left"]:self.param_box[f"{nameLine}_lane_right"]] \
					= binary_image[self.lane_limit_up:,self.param_box[f"{nameLine}_lane_left"]:self.param_box[f"{nameLine}_lane_right"]]  # take starting box
			for y in range(self.lane_limit_up, self.upper_limit, -1):  # find line's pixels (from 244 to 165)
				for x in range(self.lane_frame_width):
					if binary_image[y, x] == 1:
						how_many = np.sum(self.output[nameLine][y:y + self.max_dist, x - self.max_dist // 2:x + self.max_dist // 2])
						if how_many > self.nb_close_to:
							self.output[nameLine][y, x] = 1
							# Find de extremum of the line 
							self.output[f"y_min_{nameLine}"] = y
							# if x < self.output[f"x_min_{nameLine}"]:
							# 	self.output[f"x_min_{nameLine}"] = x
							# elif x > self.output[f"x_max_{nameLine}"]:
							# 	self.output[f"x_max_{nameLine}"] = x
			# Registering in function of the data range
			if self.upper_limit <= self.output[f"y_min_{nameLine}"] <= 194: # self.lane_limit_up-self.output[f"y_min_{nameLine}"] > 50
				self.output[f"bool_{nameLine}"] = True
				try:
					if self.disappeared[nameLine] is not None:
						self.disappeared[nameLine] = 0
				except :
					pass
			else:
				try:
					if self.disappeared[nameLine] is None:
						pass
					else:
						self.disappeared[nameLine] += 1
						if self.disappeared[nameLine] > self.bufferSize:
							self.deregister(nameLine)
				# SI la ligne n'a jamais été detectée alors on ne peut pas la supprimer.
				except :
					pass

		else:
			# Si on a pas suffisement de data, on ne peut pas compute une nouvelle ligne.
			# On incrémente alors le conpteur de disparition
			try:
				if self.disappeared[nameLine] is None:
					pass
				else:
					self.disappeared[nameLine] += 1
					if self.disappeared[nameLine] > self.bufferSize:
						self.deregister(nameLine)
			# SI la ligne n'a jamais été detectée alors on ne peut pas la supprimer.
			except :
				pass

	# Calcul les coef de notre ligne en fonction de la sortie de LaneNet
	def computePoly(self, nameLine):
		# La ligne n'est pas détectée mais existe dans le buffer. On trace la ligne n-1.
		if  self.output[f"bool_{nameLine}"] == False and self.disappeared[nameLine] is not None: 
			fit_param = self.lines[nameLine][0]
		# La ligne n'est pas détectée et n'existe pas dans le buffer
		elif self.output[f"bool_{nameLine}"] == False and self.disappeared[nameLine] is None:
			fit_param = self.fit_default[nameLine]
		# La ligne est détectée
		else:
			plot_x = np.nonzero(self.output[nameLine])[0]
			plot_y = np.nonzero(self.output[nameLine])[1]
			fit_param = np.polyfit(plot_x, plot_y, self.deg_poly, full=True)[0]
		# Register new Trace ou Mise a jour des donnees de tracking
		if self.lines[nameLine] is None:
			self.register([nameLine, [fit_param]])
		else:
			# Intégration des derniers paramètres calculés
			self.maj_plot_lines(nameLine, fit_param)
			# Merge des données
			self.mergeLine(nameLine) # Engendre des erreurs

	# Permet de récupérer les points de la ligne à partir des lignes précédentes
	def mergeLine(self, nameLine):
		# TEST UNPOND #
		if len(self.lines[nameLine]) > 2:
			# print(f"========== Merge des lignes {nameLine} ==========")
			# print(f"--Avant merge : {self.lines[nameLine][0]}--")
			for coef in range((self.deg_poly+1)):
				x = 0
				for line in self.lines[nameLine]:
					x = x + line[coef]
				self.lines[nameLine][0][coef] = round(x / len(self.lines[nameLine]), 5)
			# print(f"--Apres merge : {self.lines[nameLine][0]}--")

	# Calcul les valeurs de x en fonction de y pour une fonction donnee
	def computeTrace(self, nameLine, default=False):
		# Les lignes par default sont tracées sur une longueur moins importantes
		if default:
			self.output[f"{nameLine}_plot_y"] = np.arange(self.upper_limit_default + 2, self.lane_frame_height - 2, 1, dtype=np.float)
		else:
			self.output[f"{nameLine}_plot_y"] = np.arange(self.upper_limit + 2, self.lane_frame_height - 2, 1, dtype=np.float)
		self.output[f"{nameLine}_plot_x"] = get_x_onfit(self.lines[nameLine][0], self.output[f"{nameLine}_plot_y"], raw_scale=False)

	# Mets a jour les informations des lignes
	def update(self, binary_image, source_image):
		frame_width = binary_image.shape[1]
		self.output = {
			# Left param 
			"left": np.zeros_like(binary_image),
			"left_plot_x": None,
			"left_plot_y": None,
			"bool_left": False,
			"y_min_left": 0,
			"x_min_left": self.lane_frame_width,
			"x_max_left": 0,
			# Right param
			"right": np.zeros_like(binary_image),
			"right_plot_x": None,
			"right_plot_y": None,
			"bool_right": False,
			"y_min_right": 0,
			"x_min_right": self.lane_frame_width,
			"x_max_right": 0,
		}
		# Recherche des lignes en sortie du lanenet
		self.initSearch("left", binary_image)
		self.initSearch("right", binary_image)
		# La ligne ne droite n'est pas détectée et n'a jamais été détectée ou # Droite pas détectée sur n mais tracking actif
		if (not self.output["bool_right"] and self.lines["right"] is None) or (not self.output["bool_right"] and self.lines["right"] is not None):
			self.computePoly("right")
			self.computeTrace("right", default=True)
			# Ligne gauche n'est pas détectée et n'a jamais été détectée ou # Utilisation du tracking de la left
			if (not self.output["bool_left"] and self.lines["left"] is None) or (not self.output["bool_left"] and (self.lines["left"] is not None)):
				self.computePoly("left")
				self.computeTrace("left", default=True)
			# premiere détection hors line tracking
			elif (self.output["bool_left"] and self.lines["left"] is None)  or (self.output["bool_left"] and (self.lines["left"] is not None)):
				self.computePoly("left")
				self.computeTrace("left")
		# Ligne droite premiere détection hors line tracking ou # Détection n + Tracking droite
		elif (self.output["bool_right"] and self.lines["right"] is None) or (self.output["bool_right"] and self.lines["right"] is not None):
			self.computePoly("right")
			self.computeTrace("right")
			# Ligne gauche n'est pas détectée et n'a jamais été détectée ou # Utilisation du tracking de la left
			if (not self.output["bool_left"] and self.lines["left"] is None) or (not self.output["bool_left"] and (self.lines["left"] is not None)):
				self.computePoly("left")
				self.computeTrace("left", default=True)
			# premiere détection hors line tracking
			elif (self.output["bool_left"] and self.lines["left"] is None)  or (self.output["bool_left"] and (self.lines["left"] is not None)):
				self.computePoly("left")
				self.computeTrace("left")
		return source_image, self.output["right"], self.output["left"]