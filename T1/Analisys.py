from MachineLearn.Classes import Experiment

oExp1 = Experiment.load("Objects/EXP01_PS_20.gzip".format())

oExp2 = Experiment.load("Objects/EXP02_PS_20.gzip".format())
print(oExp1)
print(oExp2)
