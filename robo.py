
from roboflow import Roboflow
rf = Roboflow(api_key="3GSc6HlJbhrcP1cN3Mz4")
project = rf.workspace("yolov5-6agzx").project("waste-management-ql8tg")
dataset = project.version(3).download("yolov5")
