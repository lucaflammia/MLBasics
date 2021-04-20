from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk

from analisi_FIR import GetFileInfo

Image.MAX_IMAGE_PIXELS = 1000000000


class ScrollableImage(tk.Frame):
    def __init__(self, master=None, **kw):
        self.image = kw.pop('image', None)
        sw = kw.pop('scrollbarwidth', 10)
        super(ScrollableImage, self).__init__(master=master, **kw)
        self.cnvs = tk.Canvas(self, highlightthickness=0, **kw)
        self.cnvs.create_image(0, 0, anchor='nw', image=self.image)
        # Vertical and Horizontal scrollbars
        self.v_scroll = tk.Scrollbar(self, orient='vertical', width=sw)
        self.h_scroll = tk.Scrollbar(self, orient='horizontal', width=sw)
        # Grid and configure weight.
        self.cnvs.grid(row=0, column=0, sticky='nsew')
        self.h_scroll.grid(row=1, column=0, sticky='ew')
        self.v_scroll.grid(row=0, column=1, sticky='ns')
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        # Set the scrollbars to the canvas
        self.cnvs.config(xscrollcommand=self.h_scroll.set,
                         yscrollcommand=self.v_scroll.set)
        # Set canvas view to the scrollbars
        self.v_scroll.config(command=self.cnvs.yview)
        self.h_scroll.config(command=self.cnvs.xview)
        # Assign the region to be scrolled
        self.cnvs.config(scrollregion=self.cnvs.bbox('all'))
        self.cnvs.bind_class(self.cnvs, "<MouseWheel>", self.mouse_scroll)

    def mouse_scroll(self, evt):
        if evt.state == 0:
            # self.cnvs.yview_scroll(-1 * (evt.delta), 'units')  # For MacOS
            self.cnvs.yview_scroll(int(-1 * (evt.delta / 120)), 'units')  # For windows
        if evt.state == 1:
            # self.cnvs.xview_scroll(-1 * (evt.delta), 'units')  # For MacOS
            self.cnvs.xview_scroll(int(-1 * (evt.delta / 120)), 'units')  # For windows


class ScrolledCanvas(tk.Frame):
    def __init__(self, parent=None):
        Frame.__init__(self, parent)
        self.master.title("Spectrogram Viewer")
        self.pack(expand=YES, fill=BOTH)
        canv = tk.Canvas(self, relief=SUNKEN)
        canv.config(width=400, height=200)
        canv.config(highlightthickness=0)

        sbarV = Scrollbar(self, orient=VERTICAL)
        sbarH = Scrollbar(self, orient=HORIZONTAL)

        sbarV.config(command=canv.yview)
        sbarH.config(command=canv.xview)

        canv.config(yscrollcommand=sbarV.set)
        canv.config(xscrollcommand=sbarH.set)

        sbarV.pack(side=RIGHT, fill=Y)
        sbarH.pack(side=BOTTOM, fill=X)

        canv.pack(side=LEFT, expand=YES, fill=BOTH)
        self.im = Image.open("./1hr_original.jpg")
        width, height = self.im.size
        canv.config(scrollregion=(0, 0, width, height))
        self.im2 = ImageTk.PhotoImage(self.im)
        self.imgtag = canv.create_image(0, 0, anchor="nw", image=self.im2)


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.filepath = ''
        self.filename = ''

    def create_canvas(self, canvas_width, canvas_height):
        global canvas  # described as global because used outside class
        canvas = tk.Canvas(self.master, bg='papaya whip', width=canvas_width, height=canvas_height)

    def application_intro(self):
        print("APP ABILITATA")
        # restart_program_button = tk.Button(canvas, text="NUOVA ANALISI", font='Helvetica 12 bold', width=20, height=2,
        #                                    command=self.restart)
        start_program_button = tk.Button(canvas, text="SELEZIONA FIR", font='Helvetica 12 bold',
                                         width=20, height=2, command=self.start_program)
        get_text_button = tk.Button(canvas, text="ESTRAI INFO", font='Helvetica 12 bold',
                                    width=20, height=2, command=self.get_text)
        canvas.create_text(80, 20, text="ANALISI FIR", font='Helvetica 16 bold')
        # canvas.create_window(520, 200, window=restart_program_button)
        canvas.create_window(220, 200, window=start_program_button)
        canvas.create_window(520, 200, window=get_text_button)
        canvas.pack()

    def get_text(self):
        # try:

        # Toplevel object which will
        # be treated as a new window
        window = Toplevel(master)

        # sets the title of the
        # Toplevel widget
        window.title("DETTAGLI FIR")

        # sets the geometry of toplevel
        window.geometry("600x300")

        info = GetFileInfo(master, self.filepath)
        info.find_info()
        print('TIPOLOGIA FIR : {}'.format(info.tipologia))

        fact = """
        FILE : "{0}"
        TIPOLOGIA FIR : "{1}"
        PRODUTTORE : "{2}"
        DESTINATARIO : "{3}"
        TRASPORTATORE : "{4}"
        """.format(self.filename, info.tipologia, info.produttore, info.raccoglitore, info.trasportatore)

        # A Label widget to show in toplevel
        Label(window, text=fact).pack()
        tk.Button(window, text="Esci", command=window.destroy).pack()
        #
        # except AttributeError:
        #     print('NESSUN FIR SELEZIONATO')
        #     raise Exception

    def start_program(self):
        info = GetFileInfo(master)
        info.search_image()
        if not info.file:
            print('FILE NON SELEZIONATO. RIPROVA.')
            return
        self.filepath = info.file
        self.filename = info.file_only

        window = Toplevel(master)


        # sets the title of the
        # Toplevel widget
        window.title("FIR {}".format(self.filename))

        # sets the geometry of toplevel
        window.geometry("850x450+450-60")
        # img = PhotoImage(file=self.filepath)
        image = Image.open(self.filepath)
        resize_image = image.resize((800, 350))
        img = ImageTk.PhotoImage(resize_image)
        Label(window, image=img).pack()
        tk.Button(window, text="Esci", command=window.destroy).pack()

        # window = ScrollableImage(image=img, scrollbarwidth=6, width=200, height=200)
        # Label(window).pack()

        master.mainloop()

    def restart(self):
        return


master = tk.Tk()
w_main_win = 700
h_main_win = 300
master.geometry('{0}x{1}+5+5'.format(w_main_win, h_main_win))
app = Application(master=master)
app.create_canvas(w_main_win, h_main_win)

app.application_intro()
master.mainloop()

