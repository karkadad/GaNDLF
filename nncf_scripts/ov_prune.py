import sys

from openvino.frontend import FrontEndManager
from openvino.offline_transformations import apply_pruning_transformation
from openvino.offline_transformations import serialize
print(f'input model path {sys.argv[1]} \n output model path {sys.argv[2]}.xml \n output model weights {sys.argv[2]}.bin')

input_model = sys.argv[1]
fem = FrontEndManager()

def read_model(path_to_xml):
    fe = fem.load_by_framework(framework="ir")
    function = fe.convert(fe.load(path_to_xml))
    return function

func = read_model(input_model)
apply_pruning_transformation(func)
serialize(func, sys.argv[2] + '.xml', sys.argv[2] + '.bin')
print('Done!')