from PyQt5 import QtCore, QtGui, QtWidgets
from bandit_world.Ui_basic_casino import Ui_MainWindow

def pulled(ui, bandit_index):
        ui.tableWidget.item[bandit_index] += 1
        

#handler for the signal aka slot
def pull_bandit1():
    pull_bandit(0)
    complete_round()

def pull_bandit2():
    pull_bandit(1)
    complete_round()

def pull_bandit3():
    pull_bandit(2)
    complete_round()

def pull_bandit4():
    pull_bandit(3)
    complete_round()

def pull_bandit5():
    pull_bandit(4)
    complete_round()

def complete_round():
    pulls = int(ui.lbl_pull_count.value())
    pulls += 1
    ui.lbl_pull_count.display(str(pulls))


def pull_bandit(index):
    pulls = int(ui.tableWidget.item(index, 0).text())
    pulls += 1
    ui.tableWidget.item(index, 0).setText(str(pulls))



def init(table):
    for column_index in range(4):
        init_column_to_zero(table, column_index)
    
def init_column_to_zero(table, column_index):
    for i in range(5):
        item = QtWidgets.QTableWidgetItem()
        item.setText("0")
        ui.tableWidget.setItem(i, column_index, item)

def connect_buttons(ui):
    ui.pull_arm1.clicked.connect(pull_bandit1)
    ui.pull_arm2.clicked.connect(pull_bandit2)
    ui.pull_arm3.clicked.connect(pull_bandit3)
    ui.pull_arm4.clicked.connect(pull_bandit4)
    ui.pull_arm4_2.clicked.connect(pull_bandit5)
    

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
   
    init(ui.tableWidget)
    connect_buttons(ui)
    print(ui.tableWidget.item(0, 0).text())
    MainWindow.show()
    sys.exit(app.exec_())