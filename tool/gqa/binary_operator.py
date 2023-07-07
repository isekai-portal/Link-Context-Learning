from typing import Callable, Dict, Any
from root import CANT_INFER_RESULT

# [logic, logic] -> logic
logic_logic_to_logic = {
    'and',
    'or',
}

# [obj, obj] -> obj
obj_obj_to_obj = {
    'choose healthier',
    'choose larger',
    'choose less healthy',
    'choose lower',
    'choose older',
    'choose shorter',
    'choose smaller',
    'choose taller',
    'choose younger',
}

# [obj, obj] -> logic
obj_obj_to_logic = {
    'different color',
    'different shape',
    'same color',
    'same material',
    'same shape'
}

# [obj, obj] -> common attr
obj_obj_to_attr = {
    'common',
}


class Gqa2CoTBinaryMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        subtype2strfunc = [
            (logic_logic_to_logic, self.logic_logic_to_logic_str_func),
            (obj_obj_to_obj, self.obj_obj_to_obj_str_func),
            (obj_obj_to_logic, self.obj_obj_to_logic_str_func),
            (obj_obj_to_attr, self.obj_obj_to_attr_str_func),
        ]
        # merge all binary_operators
        self.binary_operators2strfunc = {}
        for subtype, strfunc in subtype2strfunc:
            for item in subtype:
                self.binary_operators2strfunc[item] = strfunc
        # only for type hint
        self.operations = {}
        self.res = []

    def logic_logic_to_logic_str_func(self, idx, arg_res):
        if self.operations[idx]['operation'] == 'and':
            cot = "The question ask about 'and' relation."
        elif self.operations[idx]['operation'] == 'or':
            cot = "The question ask about 'or' relation."
        else:
            assert False
        self.res.append(cot)
        return CANT_INFER_RESULT

    def obj_obj_to_obj_str_func(self, idx, arg_res):
        op = self.operations[idx]['operation']
        v = op.replace('choose', '').strip()
        cot = f'The question ask which one is {v} among the two objects.'
        self.res.append(cot)
        return CANT_INFER_RESULT

    def obj_obj_to_logic_str_func(self, idx, arg_res):
        op = self.operations[idx]['operation']
        v = op
        cot = f'The question ask if the two objects has {v}.'
        self.res.append(cot)
        return CANT_INFER_RESULT

    def obj_obj_to_attr_str_func(self, idx, arg_res):
        op = self.operations[idx]['operation']
        assert op == 'common'
        cot = f'The question ask the common attribute of the two objects.'
        self.res.append(cot)
        return CANT_INFER_RESULT

    def process_binary_op(self, idx, arg_res):
        assert idx == len(self.operations) - 1  # binary operator always is the last step in function program in gqa.
        assert self.operations[idx]['operation'] in self.binary_operators2strfunc, f"what's this? {self.operations[idx]['operation']}"
        return self.binary_operators2strfunc[self.operations[idx]['operation']](idx, arg_res)
