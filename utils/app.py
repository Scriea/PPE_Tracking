from tkinter import *
from typing import Optional, Tuple, Union
import customtkinter
import time
from utils.detector import Detector
from utils.general import get_stream_list
import cv2
import PIL
from PIL import ImageTk, Image

customtkinter.set_appearance_mode('System')
customtkinter.set_default_color_theme('green')


class VideoFrame(customtkinter.CTkFrame):
    def __init__(self, master, detector, tracker,  width=1024, height = 720):
        super().__init__(master, width= 1200, height=10)
        self.detector = detector
        self.tracker = tracker
        self.grid_columnconfigure(0, weight=1)
        self.title = customtkinter.CTkLabel(self, text= "Stream/Video", corner_radius=6)
        self.title.grid(row = 0, column = 0, padx = 10, pady = (10,0), sticky = "ew")

        self.canvas = customtkinter.CTkCanvas(self, width= width, height= height)
        self.canvas.grid(row= 1, column=0, sticky="news")

    def update(self):
        ret, frame = self.detector.predict()
        if ret:
            self.photo = ImageTk.PhotoImage(image =  Image.fromarray(frame))
            self.canvas.create_image(0,0,image= self.photo, anchor = customtkinter.NW)
    
    def get_something(self):
        pass


class DropMenu(customtkinter.CTkFrame):
    def __init__(self, master, detector, source= None):
        super().__init__(master, width= 800)
        self.detector = detector
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((1,3,4), weight= 3)
        self.grid_rowconfigure((0,2), weight=1)
        self.stream_list = get_stream_list(source)
        self.menu = customtkinter.StringVar(value= "")
        #self.drop = customtkinter.CTkOptionMenu(self, width= 400 ,values= self.stream_list, variable= self.menu, command=self.onMenuClick, dynamic_resizing= False)
        self.drop = customtkinter.CTkComboBox(self, width= 400, values= self.stream_list, variable= self.menu, command=self.onMenuClick, justify= "center")
        self.drop.grid(row=1, column= 0, sticky = "ew",padx = 10, pady = (20, 20))
    
    def onMenuClick(self, args):
        self.menu.set(args)
        self.detector.change_stream(str(args))

    def onButtonClick(self):
        # Returns entered arguement
        pass
    


class EntryMenu(customtkinter.CTkFrame):
    def __init__(self, master, detector):
        super().__init__(master, width= 800)
        self.grid_columnconfigure(0, weight=1)
        self.detector = detector

        self.entry = customtkinter.CTkEntry(self)
        self.entry.grid(row= 1, column=0, padx= 10, pady= 10, sticky = "ew") 
        self.entrybutton = customtkinter.CTkButton(self, text= "Play", command= self.onButtonClick)
        self.entrybutton.grid(row = 2, column =0, padx = 10, pady  =(10,10), sticky = "ews")

    def onButtonClick(self):
        # Returns entered arguement
        pass


class DropEntryFrame(customtkinter.CTkFrame):
    def __init__(self, master, detector, source = None):
        super().__init__(master)
        self.source = source
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0,1,3,4), weight= 1)
        self.grid_rowconfigure(2, weight=10)

        self.droptitle = customtkinter.CTkLabel(self, text= "List of Stream", corner_radius=6, fg_color= ["#3a7ebf", "#1f538d"])
        self.droptitle.grid(row= 0, column= 0, padx= 10, pady= (20,10), sticky= "new")

        self.dropmenu = DropMenu(self, detector= detector, source= self.source)
        self.dropmenu.grid(column = 0, row = 1, padx = 10, pady = (20, 40), sticky = "ew")


        self.entry_label = customtkinter.CTkLabel(self, text="Enter Stream URL or Path to File",fg_color= ["#3a7ebf", "#1f538d"])
        self.entry_label.grid(row=3, column= 0, sticky="ew", padx = 10, pady= (20,10) )

        self.entyrmenu = EntryMenu(self, detector= detector)
        self.entyrmenu.grid(column=0, row =4, padx = 10, pady = (20, 10), sticky = "sew")
        
    def onButtonClick(self):
        # Returns entered arguement
        pass


class App(customtkinter.CTk):
    def __init__(self, detector, tracker, window_title = "Hello World", delay = 1, source = None,) -> None:
        super().__init__()
        # self.tracker = tracker
        # self.detector = detector
        self.delay = delay
        self.source= source
        self.title(window_title)
        #self.geometry(f"1400x900")

        self.grid_columnconfigure(0, weight= 1)
        self.grid_columnconfigure(1, weight= 1)
        self.grid_rowconfigure(0, weight=1)

        # Video Widget
        self.width, self.height = detector.get_frame_size()
        self.videoframe = VideoFrame(self, detector= detector, tracker = tracker, width= self.width, height=self.height)
        self.videoframe.grid(column=0, row=0, padx = 10, pady= (20,20), sticky= 'nw')

        ## Drop-Down Menu & Source Entry Widget
        self.dropentrymenu = DropEntryFrame(self, detector= detector, source= self.source)
        self.dropentrymenu.grid(column = 1, row = 0, padx = 10, pady= (20,20), sticky = "new")


    def update(self):
        self.videoframe.update()
        self.after(self.delay, self.update)


