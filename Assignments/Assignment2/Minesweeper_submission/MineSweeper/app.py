from tkinter import *
from tkinter import messagebox

from mine_sweeper import *

GAME_STATUS = 0
MAP_COLS = 0
MAP_ROWS = 0
MINE_NUM = 0


class Application(Frame):
    def __init__(self, master=None, imagesDic=None):
        Frame.__init__(self, master)
        self.master = master
        self.grid()
        self.numofmine = 0
        self.cols = 1
        self.rows = 1
        self.squareSize = 20
        self.canvasWidth = self.cols * self.squareSize
        self.canvasHeight = self.rows * self.squareSize
        self.images = imagesDic
        for row in range(2):
            self.master.rowconfigure(row, weight=1)
        for col in range(2):
            self.master.columnconfigure(col, weight=1)

        self.map = np.zeros((1, 1))  # the map for storing whether there is mine in each block
        self.digits = np.zeros((1, 1))  # the map for storing the numbers of adjacent mines
        self.reveal = np.zeros((1, 1))  # the map for the revealed situation

        self.Frame1 = Frame(master)
        self.Frame2 = Frame(master)
        self.Frame3 = Frame(master)
        self.canvas = Canvas(self.Frame2, background="white", width=self.cols * self.squareSize,
                             height=self.rows * self.squareSize)

        self.Frame1.grid(row=0, column=0, sticky=W + E + N + S)
        self.Frame2.grid(row=0, column=1, sticky=W + E + N + S)
        self.Frame3.grid(row=1, column=1, sticky=W + E + N + S)
        self.canvas.pack(side=TOP, padx=5, pady=2)

        # widgets for Frame3
        self.label1 = Label(self.Frame3, text='rows')
        self.label1.grid(row=0, column=0)
        self.entry1 = Entry(self.Frame3)
        self.entry1.grid(row=0, column=1)
        self.label2 = Label(self.Frame3, text='cols')
        self.label2.grid(row=1, column=0)
        self.entry2 = Entry(self.Frame3)
        self.entry2.grid(row=1, column=1)
        self.label3 = Label(self.Frame3, text='num')
        self.label3.grid(row=0, column=2)
        self.entry3 = Entry(self.Frame3)
        self.entry3.grid(row=0, column=3)
        self.genButton = Button(self.Frame3, text='New Game', command=self.genMap)
        self.genButton.grid(row=0, column=4)
        '''
        self.label4 = Label(self.Frame3, text='query')
        self.label4.grid(row=1, column=2)
        self.entry4 = Entry(self.Frame3)
        self.entry4.grid(row=1, column=3)
        '''
        self.enterButton = Button(self.Frame3, text="Enter", command=self.makeMove)
        self.enterButton.grid(row=1, column=4)
        self.quitButton = Button(self.Frame3, text="Exit", command=self.quit)
        self.quitButton.grid(row=2, column=1)

    def genMap(self):
        r = self.entry1.get()
        c = self.entry2.get()
        n = self.entry3.get()
        if r == '' or c == '' or n == '':
            messagebox.showerror("Error", "Please enter parameters.")
        else:
            self.cols = int(c)
            self.rows = int(r)
            self.numofmine = int(n)
            MAP_COLS = self.cols
            MAP_ROWS = self.rows
            MINE_NUM = self.numofmine
            self.buttonlist = []
            self.map = mine_generate(self.rows, self.cols, self.numofmine)
            '''
            test case
            self.map = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
            '''
            # print(self.map)
            self.digits = np.zeros((self.rows, self.cols))
            self.reveal = np.zeros((self.rows, self.cols))
            self.visited = np.zeros((self.rows, self.cols))

            for i in range(self.map.shape[0]):
                for j in range(self.map.shape[1]):
                    self.reveal[i][j] = -3
                    if self.map[i][j] == 1:
                        if i - 1 >= 0:
                            self.digits[i - 1][j] += 1
                        if j - 1 >= 0:
                            self.digits[i][j - 1] += 1
                        if j + 1 < self.map.shape[1]:
                            self.digits[i][j + 1] += 1
                        if i + 1 < self.map.shape[0]:
                            self.digits[i + 1][j] += 1
                        if i - 1 >= 0 and j - 1 >= 0:
                            self.digits[i - 1][j - 1] += 1
                        if i - 1 >= 0 and j + 1 < self.map.shape[1]:
                            self.digits[i - 1][j + 1] += 1
                        if i + 1 < self.map.shape[0] and j - 1 >= 0:
                            self.digits[i + 1][j - 1] += 1
                        if i + 1 < self.map.shape[0] and j + 1 < self.map.shape[1]:
                            self.digits[i + 1][j + 1] += 1

            for i in range(self.map.shape[0]):
                for j in range(self.map.shape[1]):
                    if self.map[i][j] == 1:
                        self.digits[i][j] = -1
        self.drawBoard()

        self.play()

    def drawBoard(self):

        # draw the map in Frame2
        # print("redraw")
        self.canvas.delete('rec1')
        self.canvas.delete('rec2')
        newCanvasHeight = self.rows * self.squareSize
        newCanvasWidth = self.cols * self.squareSize

        newWindowHeight = int(1.5 * newCanvasHeight)
        newWindowWidth = 3 * newCanvasWidth
        self.master.update()
        # self.master.geometry(str(newWindowWidth)+"x"+str(newWindowHeight))
        # self.Frame1.config(width=newCanvasWidth, height=newCanvasHeight)

        self.canvas.config(width=newCanvasWidth, height=newCanvasHeight)
        self.Frame1.destroy()
        self.Frame1 = Frame(self.master)
        self.Frame1.grid(row=0, column=0, sticky=W + E + N + S)

        for i in range(self.rows):
            tmp_list = []
            for j in range(self.cols):
                number = int(self.digits[i][j])
                number = str(number)
                # print(number)
                img = self.images[number]
                x1 = j * self.squareSize
                y1 = i * self.squareSize
                x2 = x1 + self.squareSize
                y2 = y1 + self.squareSize
                anchor_x = (x1 + x2) / 2
                anchor_y = (y1 + y2) / 2
                self.canvas.create_image(anchor_x, anchor_y, image=img, tags='rec1')

                # mine_exist = self.map[i][j]
                # if mine_exist == 1:
                #     img2 = self.images['BOMB_IMAGE']
                #     self.canvas.create_image(anchor_x, anchor_y, image=img2, tags='rec2')

                img3 = self.images['0']
                button = Button(self.Frame1, width=12, height=12, image=img3, command=self.refresh)
                button.grid(row=i, column=j, sticky=W + E + S + N)
                button.bind('<Button-1>', lambda event, x=i, y=j: self.updateDigits(x, y))

                tmp_list.append(button)
            self.buttonlist.append(tmp_list)
        self.Frame1.grid(row=0, column=0, sticky=W + E + N + S)

    def refresh(self):

        for i in range(self.rows):
            for j in range(self.cols):

                reveal_status = self.reveal[i][j]

                if reveal_status == -1:  # step on bomb
                    img3 = self.images['HIT_BOMB_IMAGE']
                    self.buttonlist[i][j].config(image=img3)

                elif reveal_status == -2:  # set a flag
                    img3 = self.images['FLAG_IMAGE']
                    self.buttonlist[i][j].config(image=img3)

                elif reveal_status == -3:  # unrevealed
                    img3 = self.images['BLANK_IMAGE']
                    self.buttonlist[i][j].config(image=img3)
                else:  # revealed but no mine
                    num = int(reveal_status)
                    num = str(num)
                    # print(num)
                    img3 = self.images[num]
                    self.buttonlist[i][j].config(relief=SUNKEN, image=img3)

    def updateDigits(self, x, y):

        self.reveal[x][y] = self.digits[x][y]
        if self.digits[x][y] == 0:
            self.expand(x, y)

        # print(self.reveal)
        # self.refresh()

    def expand(self, x, y):
        # print(x,y)
        self.visited[x][y] = 1
        if self.digits[x][y] == 0:
            self.reveal[x][y] = 0
            if x - 1 >= 0 and self.visited[x - 1][y] == 0:
                self.visited[x - 1][y] = 1
                self.expand(x - 1, y)
            if x + 1 < self.rows and self.visited[x + 1][y] == 0:
                self.visited[x + 1][y] = 1
                self.expand(x + 1, y)
            if y - 1 >= 0 and self.visited[x][y - 1] == 0:
                self.visited[x][y - 1] = 1
                self.expand(x, y - 1)
            if y + 1 < self.cols and self.visited[x][y + 1] == 0:
                self.visited[x][y + 1] = 1
                self.expand(x, y + 1)
        else:
            self.reveal[x][y] = self.digits[x][y]
            return

    def play(self):
        self.minesweeper = MineSweeper(self.rows, self.cols, self.numofmine)
        self.query_x, self.query_y = self.minesweeper.step()
        print("query:[%d,%d]" % (self.query_x, self.query_y))
        self.updateDigits(self.query_x, self.query_y)
        self.buttonlist[self.query_x][self.query_y].invoke()
        self.Frame1.update()

    def makeMove(self):

        ##  Part 1: Answer last query and make move
        # query = self.entry4.get()
        # query=int(query)
        # print("query:%d" % (query))

        receive_prob = 1

        if self.minesweeper.cells[self.query_x][self.query_y].mined != 1:
            self.minesweeper.cells[self.query_x][self.query_y].reveal = self.digits[self.query_x][self.query_y]
            # self.minesweeper.cells[self.query_x][self.query_y].reveal = query
            if random.random() < receive_prob:
                self.minesweeper.clues_get(self.query_x, self.query_y)
            self.minesweeper.influence_chains(self.query_x, self.query_y)

        ### Part 2: Make new query and check game status
        print("clear cells:")
        print(self.minesweeper.clear_cell)
        # print("unknown cells:")
        # print(self.minesweeper.unknown_cell)
        if len(self.minesweeper.unknown_cell) > 0:
            self.query_x, self.query_y = self.minesweeper.step()
        print("query:[%d,%d]" % (self.query_x, self.query_y))
        # print("map status:%d" % (self.map[self.query_x][self.query_y]))

        self.updateDigits(self.query_x, self.query_y)
        print("mines found:")
        for m in self.minesweeper.mine_cell:
            print(m)
            self.reveal[m[0]][m[1]] = -2
        self.buttonlist[self.query_x][self.query_y].invoke()
        self.Frame1.update()

        # check if the query hits a mine
        if self.map[self.query_x][self.query_y] == 1:
            messagebox.showinfo("Warning", "Hit a mine.You Lost.")
            return

        # when there is no unknown cells
        if len(self.minesweeper.unknown_cell) == 0:
            # check if all the marked mines are correct
            mine_result = self.minesweeper.mine_cell
            if len(mine_result) != self.numofmine:
                messagebox.showinfo("Warning", "You Lost.")
                return
            else:
                for mincor in mine_result:
                    x = mincor[0]
                    y = mincor[1]
                    if self.map[x][y] != 1:
                        messagebox.showinfo("Warning", "Marked wrong mine at [%d,%d]. You Lost." % (x, y))
                        return
                    messagebox.showinfo("Congrats", "You Win!")
                    return


if __name__ == '__main__':
    root = Tk()

    images = {}
    images['BLANK_IMAGE'] = PhotoImage(file="images\\blank.gif")
    images['0'] = PhotoImage(file="images\\0.gif")
    images['1'] = PhotoImage(file="images\\1.gif")
    images['2'] = PhotoImage(file="images\\2.gif")
    images['3'] = PhotoImage(file="images\\3.gif")
    images['4'] = PhotoImage(file="images\\4.gif")
    images['5'] = PhotoImage(file="images\\5.gif")
    images['6'] = PhotoImage(file="images\\6.gif")
    images['7'] = PhotoImage(file="images\\7.gif")
    images['8'] = PhotoImage(file="images\\8.gif")
    images['-1'] = PhotoImage(file="images\\mine.gif")
    images['FLAG_IMAGE'] = PhotoImage(file="images\\flag.gif")
    images['HIT_BOMB_IMAGE'] = PhotoImage(file="images\\hit_mine.gif")
    app = Application(master=root, imagesDic=images)
    app.master.title("Minesweeper")

    app.mainloop()
