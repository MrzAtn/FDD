import tkinter
import threading
import queue
import pandas as pd
from tkinter.filedialog import*
from module.interface import Interface
from module.event_detector import *

"""
Objet permettant de lier les fonctions de l'interface (bouton+affichage HIM) et de code principal.
Cet objet instancie 2 objets: 
    1. interface : Permettant de display notre interfacc.
    2. event_detector : Permettant l'analyse des vidéos de roulage.
Utilisation de Thread afin de controler plus facilement le lancement et l'arret de traitement vidéo.
"""

class ActionInterface:
    def __init__(self):
        self.interface = Interface()
        self.ed = Event_Detector()
        # Init actions Btn
        self.interface.searchBtn_Folder.bind("<Button-1>", self.searchInFinderForVideoFolder)
        self.interface.searchBtn_CSV.bind("<Button-1>", self.searchInFinderForCSV)
        self.interface.quitBtn.bind("<Button-1>", self.stop)
        self.interface.startBtn.bind("<Button-1>", self.run)
        self.interface.inter.mainloop()

    """
    Fonction permettant de récupérer la vidéo dans le finder de l'utilisateur via une pop-up.
    """
    def searchInFinderForVideoFolder(self, *args):
        self.interface.videoFolderPath.delete(0, tkinter.END)
        # On recopie le nom du chemin dans la barre blanche
        self.interface.videoFolderPath.insert(0, askdirectory())

    """
    Fonction permettant de settle le chemin du rapport de sortie dans le finder de l'utilisateur via une pop-up.
    """
    def searchInFinderForCSV(self, *args):
        self.interface.csvPath.delete(0, tkinter.END)
        # On recopie le nom du chemin dans la barre blanche
        self.interface.csvPath.insert(0, askdirectory())

    """
    Fonction permettant d'afficher la progression de traitement.
    """
    def affichage(self, frame_count, total):
        pourc = round((frame_count/total)*100, 1)
        self.interface.prog_Label.configure(text=f"Progression: {pourc}%")

    def register_Event(self, hist_event):
        print("=== Sauvegarde des données ===")
        print(hist_event)
        csvPath = self.interface.csvPath.get()
        if csvPath == "":
            path = self.ed.gpath
        else:
            path = csvPath
        for video in hist_event:
            print('register_for_video: '+video)
            dt = pd.DataFrame()
            for event_type in hist_event[video]:
                sub_dt = pd.DataFrame() # Création d'un tableau pour chaque vidéo
                for i in range(len(hist_event[video][event_type]["start_frame"])):
                    # compute time for event start
                    min_start = hist_event[video][event_type]["start_frame"][i] // (self.ed.fps*60)
                    sec_start = hist_event[video][event_type]["start_frame"][i] % (self.ed.fps*60) // self.ed.fps
                    # compute time for event stop
                    min_end = hist_event[video][event_type]["end_frame"][i] // (self.ed.fps*60)
                    sec_end = hist_event[video][event_type]["end_frame"][i] % (self.ed.fps*60) // self.ed.fps
                    sub_dt = sub_dt.append(pd.DataFrame({
                        "Event_type": event_type,
                        "Start_Frame": [hist_event[video][event_type]["start_frame"][i]],
                        "End_Frame": [hist_event[video][event_type]["end_frame"][i]],
                        "Start_Time": [str(min_start)+"m "+str(sec_start)+"s"],
                        "End_Time": [str(min_end)+"m "+str(sec_end)+"s"]
                    }))
                dt = dt.append(sub_dt)
            dt.to_csv(path+video+".csv", index=False)
        print("=> Sauvegarde terminée")


    """
    Fonction permettant de lancer le process de détection d'évènements dans les vidéos de roulage.
    Lancement d'un thread pour un controle simplifié du processus. 
    """
    def run(self, *args):
        self.q = queue.Queue()
        self.q.put(False)
        # Récupération des buttons radio + Lancement du Thread
        self.p = threading.Thread(target=self.ed.detect_events, kwargs={"action_interface":self, "stop_process_queue":self.q, "videoFolderPath":self.interface.videoFolderPath.get(), "flag_acc": self.interface.flag_acc.get(), "flag_crossing": self.interface.flag_crossing.get(), "flag_cico": self.interface.flag_cico.get(), "flag_cut":self.interface.flag_cut.get()})
        self.p.start()

    """
    Fonction permettant de stopper l'analyse de la vidéo.
    Une sauvegarde des frames déjà analysées est réalisée.
    """
    def stop(self, *args):
        # Enregistrement de données & Arret de la fonction
        self.q.put(True)