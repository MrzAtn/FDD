import tkinter

class Interface:

    def __init__(self):
        # Init obj
        self.inter = tkinter.Tk()
        self.logo = tkinter.PhotoImage(file='ressources/logo.gif')
        self.inter.resizable(width=False, height=False)
        self.inter.title("HIM Event Detector")
        self.can = tkinter.Canvas(master=self.inter)
        self.can.pack(padx=5, pady=5)
        self.flag_crossing = bool
        self.flag_cico =  bool
        self.flag_acc = bool 
        self.flag_cut = bool

        # Frame parcours données
        self.frame_d = tkinter.LabelFrame(self.can, text='Input:', padx=2, pady=5)
        self.frame_d.pack(fill='both', expand='yes')
        path_Label = tkinter.Label(master=self.frame_d, text='Video Path:')
        path_Label.grid(row=0, column=0, sticky=tkinter.W)
        self.videoFolderPath = tkinter.Entry(master=self.frame_d, width=55) # ajouter ecoute
        self.videoFolderPath.grid(row=0, column=1)
        self.searchBtn_Folder = tkinter.Button(master=self.frame_d, text='SEARCH')
        self.searchBtn_Folder.grid(row=0, column=3, padx=10)

        # Frame controle
        self.mid_Frame = tkinter.Frame(self.can)
        self.mid_Frame.pack(fill='both', expand='yes')
        
        # Sub frame for event analyse selection
        self.frame_c = tkinter.LabelFrame(self.mid_Frame, text='Analysis choices:', padx=30, pady=10)
        self.frame_c.grid(row=0, column=0, sticky=tkinter.W)
        self.flag_crossing, self.flag_cico, self.flag_acc, self.flag_cut = tkinter.IntVar(), tkinter.IntVar(), tkinter.IntVar(), tkinter.IntVar()
        self.Crossing_CB = tkinter.Checkbutton(master=self.frame_c, text='Crossing', variable=self.flag_crossing, onvalue=1, offvalue=0)
        self.Crossing_CB.grid(row=0, column=0, sticky=tkinter.W)
        self.CICO_CB = tkinter.Checkbutton(master=self.frame_c, text='CICO', variable=self.flag_cico, onvalue=1, offvalue=0)
        self.CICO_CB.grid(row=1, column=0, sticky=tkinter.W)
        self.ACC_CB = tkinter.Checkbutton(master=self.frame_c, text='ACC/NoACC', variable=self.flag_acc, onvalue=1, offvalue=0)
        self.ACC_CB.grid(row=2, column=0, sticky=tkinter.W)
        self.CUT_CB  = tkinter.Checkbutton(master=self.frame_c, text='Cut In/Out', variable=self.flag_cut, onvalue=1, offvalue=0)
        self.CUT_CB.grid(row=3, column=0, sticky=tkinter.W)
        
        # Logo
        self.canvas_logo = tkinter.Canvas(self.mid_Frame, width=310, height=130)
        self.canvas_logo.create_image(160, 80, image=self.logo)
        self.canvas_logo.grid(row=0, column=1)

        # Frame Info vidéo
        self.bot_Frame = tkinter.Frame(self.can)
        self.bot_Frame.pack(fill='both', expand='yes')
        self.frame_out = tkinter.LabelFrame(self.bot_Frame, text='Output:', padx=10, pady=28)
        self.frame_out.grid(row=0, column=0)
        path_csv = tkinter.Label(master=self.frame_out, text='CSV Path:')
        path_csv.grid(row=0, column=0, sticky=tkinter.W)
        self.csvPath = tkinter.Entry(master=self.frame_out, width=35) # ajouter ecoute
        self.csvPath.grid(row=0, column=1)
        self.searchBtn_CSV = tkinter.Button(master=self.frame_out, text='SEARCH')
        self.searchBtn_CSV.grid(row=0, column=3, padx=10)

        # Sub frame for control
        self.frame_b = tkinter.LabelFrame(self.bot_Frame, text='Control:', padx=10, pady=5)
        self.frame_b.grid(row=0, column=1)
        self.startBtn = tkinter.Button(master=self.frame_b, text='START PROCESS', width=12)
        self.startBtn.grid(row=0, column=0, sticky=tkinter.W)
        self.quitBtn = tkinter.Button(master=self.frame_b, text='STOP PROCESS', width=12)
        self.quitBtn.grid(row=1, column=0, sticky=tkinter.W)
        self.prog_Label = tkinter.Label(master=self.frame_b, text="Progression: 0%")
        self.prog_Label.grid(row=2, column=0, sticky=tkinter.W)