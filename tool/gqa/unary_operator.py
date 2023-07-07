import re

from root import PHRASE_ST_PLACEHOLDER, PHRASE_ED_PLACEHOLDER, get_box_xyxy, CANT_INFER_RESULT

obj2attr = {
    'query',
    'choose',
}
obj2logic = {}
obj2obj = {}


class Gqa2CoTUnaryMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # only for type hint
        self.operations = {}
        self.scene = {}

    def process_chain_op(self, chain, arg_res):
        # print(f"process_chain_op({chain}, {arg_res})")
        # select
        # filter - select
        # exist - filter - select
        # relate - filter - select
        # query - relate - filter - select
        assert len(chain) > 0
        op = self.operations[chain[0]]['operation']
        if 'query' == op:
            ret = self.process_attr_chain_query(chain, arg_res)
        elif 'choose' in op:
            ret = self.process_attr_chain_choose(chain, arg_res)
        elif 'relate' == op:
            ret = self.process_obj_chain_relate(chain, arg_res)
        elif 'filter' in op or 'select' in op:
            ret = self.process_obj_chain_common(chain, arg_res)
        elif 'exist' == op:
            ret = self.process_logic_chain_exist(chain, arg_res)
        elif 'verify' in op:
            ret = self.process_logic_chain_verify(chain, arg_res)
        elif op in ['different', 'same']:
            ret = self.process_logic_chain_diff_same(chain, arg_res, op)
        else:
            assert False, f"what's this? {op}"
        # print(f"res: {self.res}")
        return ret

    def process_logic_chain_diff_same(self, chain, arg_res, op_type):
        assert len(chain) > 1
        argument = self.operations[chain[0]]['argument']
        arg_res = self.process_chain_op(chain[1:], arg_res)
        rids = arg_res['rids']
        assert isinstance(rids, list), f"{arg_res}"
        assert '-' not in rids
        cot = f'Check if they have {op_type} {argument}.'
        self.res.append(cot)

    def process_attr_chain_query(self, chain, arg_res):
        assert len(chain) > 1
        assert chain[0] == len(self.operations) - 1
        assert 'query' == self.operations[chain[0]]['operation']
        argument = self.operations[chain[0]]['argument']
        # predecessor
        arg_res = self.process_chain_op(chain[1:], arg_res)
        # box and cot
        rids = arg_res['rids']
        _opdn = arg_res['name']
        assert bool(rids)
        assert isinstance(rids, (list, tuple))
        assert isinstance(rids[0], (int, str))
        # self.add_boxes_by_rids(rids)
        # cot = f"Think the {argument} of {PHRASE_ST_PLACEHOLDER}the {_opdn}{PHRASE_ED_PLACEHOLDER}."
        # self.res.append(cot)
        return CANT_INFER_RESULT

    def process_attr_chain_choose(self, chain, arg_res):
        assert len(chain) > 1
        assert chain[0] == len(self.operations) - 1
        assert 'choose' in self.operations[chain[0]]['operation']
        argument = self.operations[chain[0]]['argument']
        # predecessor
        arg_res = self.process_chain_op(chain[1:], arg_res)
        # box and cot
        rids = arg_res['rids']
        _opdn = arg_res['name']
        assert isinstance(rids, (list, tuple))
        assert isinstance(rids[0], (int, str))
        assert len(rids) == 1
        _v = argument.split(',')
        if len(_v) == 3:
            # rel with another obj
            n, a, r, p = parse_argument(argument)
            assert len(r) == 1, f"{_v}"
            _a = a.split('|')
            assert len(_a) >= 2
            a = f"{','.join(_a[:-1])} or {_a[-1]}"
            if p == 'o':
                self.add_boxes_by_rids(rids)
                self.add_boxes_by_rids(r)
                cot = f"Think {PHRASE_ST_PLACEHOLDER}the {_opdn}{PHRASE_ED_PLACEHOLDER} is {a} {PHRASE_ST_PLACEHOLDER}the {n}{PHRASE_ED_PLACEHOLDER}."
                self.res.append(cot)
            elif p == 's':
                self.add_boxes_by_rids(r)
                self.add_boxes_by_rids(rids)
                cot = f"Think {PHRASE_ST_PLACEHOLDER}the {n}{PHRASE_ED_PLACEHOLDER} is {a} {PHRASE_ST_PLACEHOLDER}the {_opdn}{PHRASE_ED_PLACEHOLDER}."
                self.res.append(cot)
            else:
                assert False
        elif len(_v) == 1:
            # attr
            _a = _v[0].split('|')
            a = f"{','.join(_a[:-1])} or {_a[-1]}"
            self.add_boxes_by_rids(rids)
            cot = f"Think {PHRASE_ST_PLACEHOLDER}the {_opdn}{PHRASE_ED_PLACEHOLDER} is {a}."
            self.res.append(cot)
        else:
            assert False, f"{_v}, {self.operations}"

    def process_logic_chain_exist(self, chain, arg_res):
        assert len(chain) > 1
        assert 'exist' == self.operations[chain[0]]['operation']
        arg_res = self.process_chain_op(chain[1:], arg_res)
        rids = arg_res['rids']
        assert isinstance(rids, list), f"{arg_res}"
        if '-' in rids:
            cot = f"It doesn't exist."
            self.res.append(cot)
        else:
            cot = f"It exists."
            self.res.append(cot)

    def process_logic_chain_verify(self, chain, arg_res):
        assert len(chain) > 1
        assert 'verify' in self.operations[chain[0]]['operation']
        argument = self.operations[chain[0]]['argument']
        arg_res = self.process_chain_op(chain[1:], arg_res)
        # box and cot
        rids = arg_res['rids']
        _opdn = arg_res['name']
        assert isinstance(rids, (list, tuple))
        assert isinstance(rids[0], (int, str))
        assert len(rids) == 1
        _v = argument.split(',')
        if len(_v) == 3:
            # rel with another obj
            n, a, r, p = parse_argument(argument)
            assert len(r) == 1, f"{_v}"
            if p == 'o':
                if r[0] == '-':
                    self.add_boxes_by_rids(rids)
                    cot = f"Verify if {PHRASE_ST_PLACEHOLDER}the {_opdn}{PHRASE_ED_PLACEHOLDER} is {a} the {n}."
                    self.res.append(cot)
                else:
                    self.add_boxes_by_rids(rids)
                    self.add_boxes_by_rids(r)
                    cot = f"Verify if {PHRASE_ST_PLACEHOLDER}the {_opdn}{PHRASE_ED_PLACEHOLDER} is {a} {PHRASE_ST_PLACEHOLDER}the {n}{PHRASE_ED_PLACEHOLDER}."
                    self.res.append(cot)
            elif p == 's':
                if r[0] == '-':
                    self.add_boxes_by_rids(rids)
                    cot = f"Verify if the {n} is {a} {PHRASE_ST_PLACEHOLDER}the {_opdn}{PHRASE_ED_PLACEHOLDER}."
                    self.res.append(cot)
                else:
                    self.add_boxes_by_rids(r)
                    self.add_boxes_by_rids(rids)
                    cot = f"Verify if {PHRASE_ST_PLACEHOLDER}the {n}{PHRASE_ED_PLACEHOLDER} is {a} {PHRASE_ST_PLACEHOLDER}the {_opdn}{PHRASE_ED_PLACEHOLDER}."
                    self.res.append(cot)
            else:
                assert False
        elif len(_v) == 1:
            # attr
            _a = _v[0]
            attr = self.operations[chain[0]]['operation'].replace('verify', '').strip()
            self.add_boxes_by_rids(rids)
            if attr:
                cot = f"Verify if the {attr} of {PHRASE_ST_PLACEHOLDER}the {_opdn}{PHRASE_ED_PLACEHOLDER} is {_a}."
            else:
                cot = f"Verify if {PHRASE_ST_PLACEHOLDER}the {_opdn}{PHRASE_ED_PLACEHOLDER} is {_a}."
            self.res.append(cot)
        else:
            assert False, f"{_v}, {self.operations}"

    def process_obj_chain_relate(self, chain, arg_res, name_prefix=''):
        assert len(chain) > 1
        assert 'relate' == self.operations[chain[0]]['operation']
        argument = self.operations[chain[0]]['argument']
        arg_res = self.process_chain_op(chain[1:], arg_res)
        # box and cot
        rids = arg_res['rids']
        _opdn = arg_res['name']
        assert isinstance(rids, (list, tuple))
        assert isinstance(rids[0], (int, str))
        assert len(rids) == 1
        _v = argument.split(',')
        if len(_v) == 3:
            n, a, r, p = parse_argument(argument)
            assert len(r) == 1, f"{_v}"
            if p == 'o':
                if '-' not in r:
                    self.add_boxes_by_rids(r)
                    cot = f"Check the {name_prefix}{n} that it {a}, got {PHRASE_ST_PLACEHOLDER}the {n}{PHRASE_ED_PLACEHOLDER}."
                    self.res.append(cot)
                else:
                    cot = f"Check the {name_prefix}{n} that it {a}."
                    self.res.append(cot)
            elif p == 's':
                if '-' not in r:
                    self.add_boxes_by_rids(r)
                    cot = f"Check the {name_prefix}{n} {a} it, got {PHRASE_ST_PLACEHOLDER}the {n}{PHRASE_ED_PLACEHOLDER}."
                    self.res.append(cot)
                else:
                    cot = f"Check the {name_prefix}{n} {a} it."
                    self.res.append(cot)
            elif p == '_':
                if '-' not in r:
                    self.add_boxes_by_rids(r)
                    cot = f"Check the {name_prefix}{n} has the {a}, got {PHRASE_ST_PLACEHOLDER}the {n}{PHRASE_ED_PLACEHOLDER}."
                    self.res.append(cot)
                else:
                    cot = f"Check the {name_prefix}{n} has the {a}."
                    self.res.append(cot)
            else:
                assert False
            return {'rids': r, 'name': n}
        else:
            assert False

    def process_obj_chain_common(self, chain, arg_res):
        # filter and select
        idx = 0
        while idx < len(chain):
            op = self.operations[chain[idx]]['operation']
            if 'filter' not in op:
                break
            idx += 1
        assert idx != len(chain)
        if self.operations[chain[idx]]['operation'] == 'select':
            text, rids = parse_select(self.operations[chain[idx]]['argument'])
            all_filter_text = [self.operations[_]['argument'] for _ in chain[:idx]]
            if len(rids) > 0 and '-' not in rids:
                self.add_boxes_by_rids(rids)
                cot = " ".join([f"Find {PHRASE_ST_PLACEHOLDER}the", *all_filter_text, text, f"{PHRASE_ED_PLACEHOLDER}."])
                self.res.append(cot)
            else:
                cot = " ".join([f"Find the", *all_filter_text, f"{text}."])
                self.res.append(cot)
            return {'rids': rids, 'name': text}
        elif self.operations[chain[idx]]['operation'] == 'relate':
            all_filter_text = [self.operations[_]['argument'] for _ in chain[:idx]]
            name_prefix = " ".join(all_filter_text).strip()
            if name_prefix:
                name_prefix = name_prefix + ' '
            return self.process_obj_chain_relate(chain[1:], arg_res, name_prefix=name_prefix)
        else:
            assert False


TEXT_NAME_MAP = {
    'he': 'male',
    'she': 'female',
}


def parse_argument(argument):
    # some horrible code
    _v = argument.split(',')
    _name = _v[0] if _v[0] not in ['-', '_'] else 'object'
    _a = _v[1]
    _new_rids = re.findall(r'\((.*)\)', _v[2])[0].split(',')
    _pos = _v[2][0]
    assert _pos in ['o', 's', '_']
    if _name in TEXT_NAME_MAP:
        _name = TEXT_NAME_MAP[_name]
    assert len(_new_rids) == 1
    return _name, _a, _new_rids, _pos


def parse_select(argument):
    _new_rids = re.findall(r'\((.*)\)', argument)[0].split(',')
    _text = re.sub(r'\((.*)\)', '', argument).strip()
    if _text in TEXT_NAME_MAP:
        _text = TEXT_NAME_MAP[_text]
    return _text, _new_rids
