import cv2
import numpy as np
import pandas as pd

# Ajouter dans le module de fonction du projet
def factorielle(n):
    res = 1
    for k in range(1, n):
        res += res * k
    return res

# Print des données à l'instant t
def displayResult(l_t, frame_count):
    print(f"=============== Tracking Lignes: frame n°{frame_count} ===============")
    print("Tracker gauche:",  l_t.disappeared["left"])
    print("Tracking gauche:")
    try:
        for line in l_t.lines["left"]:
            print(line)
        print("\n")
    except: 
        pass
    print("Tracker droite:", l_t.disappeared["right"])
    print("Tracking droite:")
    try:
        for line in l_t.lines["right"]:
            print(line)
        print("\n\n")
    except:
        pass

def maskframe(frame): # mask useless regions in image
    # Create the basic black image 
    mask_left = np.zeros(frame.shape, dtype = "uint8")
    mask_right = np.zeros(frame.shape, dtype = "uint8")
    mask_top = np.zeros(frame.shape, dtype = "uint8")
    mask_bot = np.zeros(frame.shape, dtype = "uint8")
    mask = np.zeros(frame.shape, dtype = "uint8")

    # Draw a white, filled rectangle on the mask image
    cv2.rectangle(mask_left, (0, 45), (240, 240), (255,255,255), -1)
    cv2.rectangle(mask_right, (720, 45), (475, 240), (255,255,255), -1)
    cv2.rectangle(mask_top, (0, 0), (720, 45), (255,255,255), -1)
    cv2.rectangle(mask_bot, (0, 395), (720, 576), (255,255,255), -1)
    mask = cv2.bitwise_or(mask_left, mask_right)
    mask = cv2.bitwise_or(mask, mask_top)
    mask = cv2.bitwise_or(mask, mask_bot)
    mask = cv2.bitwise_not(mask)

    # Apply the mask and display the result
    maskedImg = cv2.bitwise_and(frame, mask)
    maskedImg = maskedImg[140:395]
    
    return maskedImg

def draw_lines(source_image, dict_plot, lane_color):
    lane_frame_width = 720  # lanenet input size
    lane_frame_height = 256  # lanenet input size

    frame_width = 720  # after maskframe
    frame_height = 255  # after maskframe
    """
    Draws line on source_image according to line's equation in dict_plot and lane's color in lane_color
    Manages temp on each line, ie if a line has not be seen for a while it disapears
    """
    if lane_color[0] == 255 and dict_plot["left_temp"] > 0:
        plot_x = np.copy(dict_plot["left_plot_x"])
        plot_y = np.copy(dict_plot["left_plot_y"])
        dict_plot["left_temp"] -= 1
        if dict_plot["left_temp"] <= 0:
            dict_plot["left_plot_x"] = None
            dict_plot["left_fit_param"] = None
    elif lane_color[2] == 255 and dict_plot["right_temp"] > 0:
        plot_x = np.copy(dict_plot["right_plot_x"])
        plot_y = np.copy(dict_plot["right_plot_y"])
        dict_plot["right_temp"] -= 1
        if dict_plot["right_temp"] <= 0:
            dict_plot["right_plot_x"] = None
            dict_plot["right_fit_param"] = None
    else:
        return source_image, dict_plot
    try:
        plot_x *= frame_width/lane_frame_width
        plot_y *= frame_height/lane_frame_height
        for y in range(len(plot_y)):
            cv2.circle(source_image, (int(plot_x[y]), int(plot_y[y])), 2, lane_color, -1)
        return source_image, dict_plot
    except:
        return source_image, dict_plot

def get_x_onfit(fit_param, plot_y, raw_scale=True):
    frame_width = 720  # after maskframe
    frame_height = 255  # after maskframe
    lane_frame_width = 720  # lanenet input size
    lane_frame_height = 256  # lanenet input size
    if raw_scale:
        plot_y *= lane_frame_height/frame_height
    plot_x = fit_param[0] * (plot_y ** 2) + fit_param[1] * plot_y + fit_param[2]
    # plot_x = fit_param[0]*(plot_y**3)+fit_param[1]*(plot_y**2)+fit_param[2]*plot_y+fit_param[3]
    if raw_scale:
        return (plot_x * frame_width/lane_frame_width)
    else:
        return plot_x

def average(number1, number2):
    return (number1 + number2) / 2