import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from detect_area.count_and_mean_area_BGR import count_relation_of_mean_face_and_eye



