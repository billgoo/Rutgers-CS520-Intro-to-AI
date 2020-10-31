import numpy as np
import copy
from tkinter import *
from MapGen import maze_generate
from DFS import *
from BFS import *
from A_star import *



DARK_SQUARE_COLOR = "black"
LIGHT_SQUARE_COLOR = "white"
PATH_COLOR = "green"



class Application(Frame):
    def __init__(self, master=None, map=np.zeros((10,10))):
        Frame.__init__(self,master)
        self.size = 30
        self.rows = map.shape[0]
        self.cols = map.shape[1]
        self.width = self.cols*self.size
        self.height = self.rows*self.size
        self.canvas = Canvas(self, borderwidth=0, highlightthickness=0, background="white", width=self.width, height=self.height)
        self.canvas.pack(side="top", fill="both", expand=True, padx=2, pady=2)
        self.canvas.bind("<Configure>", self.resize)

        self.map = map
        self.createWidgets()
        self.pack(side="top",fill="both",expand="true",padx=5,pady=5)

        #self.grid(expand="true")

    def resize(self, event):
        x_size = int((event.width-1)/self.cols)
        y_size = int((event.height-1)/self.rows)
        self.size = min(x_size,y_size)
        self.canvas.delete("square")

        for i in range(self.rows):
            for j in range(self.cols):
                color = LIGHT_SQUARE_COLOR
                if self.map[i][j] == 1:
                    color = DARK_SQUARE_COLOR
                elif self.map[i][j] == 0:
                    color = LIGHT_SQUARE_COLOR
                else:
                    color = PATH_COLOR

                x1 = j*self.size
                y1 = i*self.size
                x2 = x1 + self.size
                y2 = y1 + self.size
                #print(i,j,self.map[i][j],x1,y1,x2,y2,color)
                self.canvas.create_rectangle(
                    x1, y1,
                    x2, y2,
                    outline="black", fill=color, tags="square"
                )

    def refresh(self):
        self.canvas.delete("square")
        self.rows = self.map.shape[0]
        self.cols = self.map.shape[1]
        #print(self.map)

        for i in range(self.rows):
            for j in range(self.cols):
                color = LIGHT_SQUARE_COLOR
                if self.map[i][j] == 1:
                    color = DARK_SQUARE_COLOR
                elif self.map[i][j] == 0:
                    color = LIGHT_SQUARE_COLOR
                else:
                    color = PATH_COLOR

                x1 = j*self.size
                y1 = i*self.size
                x2 = x1 + self.size
                y2 = y1 + self.size
                #print(i,j,self.map[i][j],x1,y1,x2,y2,color)
                self.canvas.create_rectangle(
                    x1, y1,
                    x2, y2,
                    outline="black", fill=color, tags="square"
                )




    def createWidgets(self):
        self.quitButton = Button(self, text = 'Quit', command = self.quit)
        self.quitButton.pack(side="bottom")
        self.dfsButton = Button(self, text='DFS', command=self.runDFS)
        self.dfsButton.pack(side="bottom")
        #self.dfsButton.bind("<Button-1>",self.refresh)
        self.bfsButton = Button(self, text='BFS', command=self.runBFS)
        self.bfsButton.pack(side="bottom")
        self.astarButton1 = Button(self, text='A*(Euclidean)', command=self.runAstarEuclidean)
        self.astarButton1.pack(side="bottom")
        self.astarButton2 = Button(self, text='A*(Manhattan)', command=self.runAstarManhattan)
        self.astarButton2.pack(side="bottom")
        self.genButton = Button(self, text='Generate', command=self.generate2)
        self.genButton.pack(side="bottom")
        #self.genButton.bind("<ButtonPress>",self.refresh)



        self.label1 = Label(self, text='dim')
        self.label1.pack(side="left")
        self.e1 = Entry(self)
        self.e1.pack(side="left")
        self.label2 = Label(self, text='q')
        self.label2.pack(side="left")
        self.e2 = Entry(self)
        self.e2.pack(side="left")

    def generate(self):
        dim = self.e1.get()
        q = self.e2.get()
        dim = int(dim)
        q = float(q)
        self.map = maze_generate(dim,q)
        #print(self.map)
        self.refresh()

    def generate2(self):
        self.map = np.load('/Users/liyunfan/Desktop/data/100_hard_maze_NODE.npy')

        self.refresh()

    def runDFS(self):
        temp = copy.deepcopy(self.map)
        solution = dfs(self.map)
        path = solution['PATH']
        print('DFS:')
        print(path)
        for point in path:
            i = point[0]
            j = point[1]
            self.map[i][j] = 2
        self.refresh()
        self.map = copy.deepcopy(temp)

    def runBFS(self):
        temp = copy.deepcopy(self.map)
        solution = bfs(self.map)
        path = solution['PATH']
        print('BFS:')
        print(path)
        for point in path:
            i = point[0]
            j = point[1]
            self.map[i][j] = 2
        self.refresh()
        self.map = copy.deepcopy(temp)

    def runAstarEuclidean(self):
        temp = copy.deepcopy(self.map)
        solution = a_star_euclidean(self.map)
        path=solution['PATH']

        if solution['NODE']==np.inf:
            path=[]
        print('A* Euclidean:')
        print(path)
        for point in path:
            i = point[0]
            j = point[1]
            self.map[i][j] = 2
        self.refresh()
        self.map = copy.deepcopy(temp)


    def runAstarManhattan(self):
        temp = copy.deepcopy(self.map)
        solution = a_star_manhattan(self.map)
        path = solution['PATH']
        if solution['NODE']==np.inf:
            path=[]
        print('A* Manhattan:')
        print(path)
        for point in path:
            i = point[0]
            j = point[1]
            self.map[i][j] = 2
        self.refresh()
        self.map = copy.deepcopy(temp)



if __name__ == '__main__':
    root = Tk()
    app = Application(root)
    app.master.title('Maze Runner')
    root.mainloop()