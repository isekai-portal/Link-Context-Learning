import numpy as np


def function_to_str(f):
    value_str = ''
    if f['value_inputs']:
        value_str = '[%s]' % ','.join(f['value_inputs'])
    return '%s%s' % (f['function'], value_str)


def tree_to_prefix(program_tree):
    output = []

    def helper(cur):
        output.append({
            'idx': cur['idx'],
            'function': cur['function'],
            'value_inputs': [x for x in cur['value_inputs']],
        })
        for node in cur['inputs']:
            helper(node)

    helper(program_tree)
    return output


def list_to_tree(program_list):
    def build_subtree(cur, idx):
        return {
            'idx': idx,
            'function': cur['function'],
            'value_inputs': [x for x in cur['value_inputs']],
            'inputs': [build_subtree(program_list[i], i) for i in cur['inputs']],
        }

    return build_subtree(program_list[-1], len(program_list) - 1)


def load_scene(s):
    table = []
    for i, o in enumerate(s['objects']):
        item = {}

        item['id'] = '%d-%d' % (s['image_index'], i)
        item['image_index'] = s['image_index']
        item['object_index'] = i
        if '3d_coords' in o:
            item['position'] = [np.dot(o['3d_coords'], s['directions']['right']),
                                np.dot(o['3d_coords'], s['directions']['front']),
                                o['3d_coords'][2]]
        else:
            item['position'] = o['position']
        item['color'] = o['color']
        item['material'] = o['material']
        item['shape'] = o['shape']
        item['size'] = o['size']
        table.append(item)
    return table


CLEVR_COLORS = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow']
CLEVR_MATERIALS = ['rubber', 'metal']
CLEVR_SHAPES = ['cube', 'cylinder', 'sphere']
CLEVR_SIZES = ['large', 'small']

CLEVR_ANSWER_CANDIDATES = {
    'count': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'equal_color': ['yes', 'no'],
    'equal_integer': ['yes', 'no'],
    'equal_material': ['yes', 'no'],
    'equal_shape': ['yes', 'no'],
    'equal_size': ['yes', 'no'],
    'exist': ['yes', 'no'],
    'greater_than': ['yes', 'no'],
    'less_than': ['yes', 'no'],
    'query_color': ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow'],
    'query_material': ['metal', 'rubber'],
    'query_size': ['small', 'large'],
    'query_shape': ['cube', 'cylinder', 'sphere'],
    'same_color': ['yes', 'no'],
    'same_material': ['yes', 'no'],
    'same_size': ['yes', 'no'],
    'same_shape': ['yes', 'no']
}


class ClevrExecutor:
    """Symbolic program executor for CLEVR"""

    def __init__(self):

        self.colors = CLEVR_COLORS
        self.materials = CLEVR_MATERIALS
        self.shapes = CLEVR_SHAPES
        self.sizes = CLEVR_SIZES
        self.answer_candidates = CLEVR_ANSWER_CANDIDATES

        self.modules = {}
        self._register_modules()

    def run_question(self, question, scene, debug=False):
        program = question['program']
        prefix = tree_to_prefix(list_to_tree(program))
        prefix2func = [{'idx': item['idx'], 'token': function_to_str(item)} for item in prefix]

        scene = load_scene(scene)

        ans, temp = None, None
        self.exe_trace = []
        for item in prefix2func[::-1]:
            token = item['token']
            if token == 'scene':
                if temp is not None:
                    ans = 'error'
                    break
                temp = ans
                ans = list(scene)
            elif token in self.modules:
                module = self.modules[token]
                if token.startswith('same') or token.startswith('relate'):
                    ans = module(ans, scene)
                else:
                    ans = module(ans, temp)
                if ans == 'error':
                    break
            else:
                assert False, f'token: {token}'
            item['ans'] = ans
            self.exe_trace.append(ans)
            if debug:
                print(token)
                print('ans:')
                self._print_debug_message(ans)
                print('temp: ')
                self._print_debug_message(temp)
                print()
        ans = str(ans)
        return ans, prefix2func

    def _print_debug_message(self, x):
        if type(x) == list:
            for o in x:
                print(self._object_info(o))
        elif type(x) == dict:
            print(self._object_info(x))
        else:
            print(x)

    def _object_info(self, obj):
        return '%s %s %s %s at %s' % (obj['size'], obj['color'], obj['material'], obj['shape'], str(obj['position']))

    def _register_modules(self):
        self.modules['count'] = self.count
        self.modules['equal_color'] = self.equal_color
        self.modules['equal_integer'] = self.equal_integer
        self.modules['equal_material'] = self.equal_material
        self.modules['equal_shape'] = self.equal_shape
        self.modules['equal_size'] = self.equal_size
        self.modules['exist'] = self.exist
        self.modules['filter_color[blue]'] = self.filter_blue
        self.modules['filter_color[brown]'] = self.filter_brown
        self.modules['filter_color[cyan]'] = self.filter_cyan
        self.modules['filter_color[gray]'] = self.filter_gray
        self.modules['filter_color[green]'] = self.filter_green
        self.modules['filter_color[purple]'] = self.filter_purple
        self.modules['filter_color[red]'] = self.filter_red
        self.modules['filter_color[yellow]'] = self.filter_yellow
        self.modules['filter_material[rubber]'] = self.filter_rubber
        self.modules['filter_material[metal]'] = self.filter_metal
        self.modules['filter_shape[cube]'] = self.filter_cube
        self.modules['filter_shape[cylinder]'] = self.filter_cylinder
        self.modules['filter_shape[sphere]'] = self.filter_sphere
        self.modules['filter_size[large]'] = self.filter_large
        self.modules['filter_size[small]'] = self.filter_small
        self.modules['greater_than'] = self.greater_than
        self.modules['less_than'] = self.less_than
        self.modules['intersect'] = self.intersect
        self.modules['query_color'] = self.query_color
        self.modules['query_material'] = self.query_material
        self.modules['query_shape'] = self.query_shape
        self.modules['query_size'] = self.query_size
        self.modules['relate[behind]'] = self.relate_behind
        self.modules['relate[front]'] = self.relate_front
        self.modules['relate[left]'] = self.relate_left
        self.modules['relate[right]'] = self.relate_right
        self.modules['same_color'] = self.same_color
        self.modules['same_material'] = self.same_material
        self.modules['same_shape'] = self.same_shape
        self.modules['same_size'] = self.same_size
        self.modules['union'] = self.union
        self.modules['unique'] = self.unique

    def count(self, scene, _):
        if type(scene) == list:
            return len(scene)
        return 'error'

    def equal_color(self, color1, color2):
        if type(color1) == str and color1 in self.colors and type(color2) == str and color2 in self.colors:
            if color1 == color2:
                return 'yes'
            else:
                return 'no'
        return 'error'

    def equal_integer(self, integer1, integer2):
        if type(integer1) == int and type(integer2) == int:
            if integer1 == integer2:
                return 'yes'
            else:
                return 'no'
        return 'error'

    def equal_material(self, material1, material2):
        if type(material1) == str and material1 in self.materials and type(material2) == str and material2 in self.materials:
            if material1 == material2:
                return 'yes'
            else:
                return 'no'
        return 'error'

    def equal_shape(self, shape1, shape2):
        if type(shape1) == str and shape1 in self.shapes and type(shape2) == str and shape2 in self.shapes:
            if shape1 == shape2:
                return 'yes'
            else:
                return 'no'
        return 'error'

    def equal_size(self, size1, size2):
        if type(size1) == str and size1 in self.sizes and type(size2) == str and size2 in self.sizes:
            if size1 == size2:
                return 'yes'
            else:
                return 'no'
        return 'error'

    def exist(self, scene, _):
        if type(scene) == list:
            if len(scene) != 0:
                return 'yes'
            else:
                return 'no'
        return 'error'

    def filter_blue(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'blue':
                    output.append(o)
            return output
        return 'error'

    def filter_brown(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'brown':
                    output.append(o)
            return output
        return 'error'

    def filter_cyan(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'cyan':
                    output.append(o)
            return output
        return 'error'

    def filter_gray(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'gray':
                    output.append(o)
            return output
        return 'error'

    def filter_green(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'green':
                    output.append(o)
            return output
        return 'error'

    def filter_purple(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'purple':
                    output.append(o)
            return output
        return 'error'

    def filter_red(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'red':
                    output.append(o)
            return output
        return 'error'

    def filter_yellow(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == 'yellow':
                    output.append(o)
            return output
        return 'error'

    def filter_rubber(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['material'] == 'rubber':
                    output.append(o)
            return output
        return 'error'

    def filter_metal(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['material'] == 'metal':
                    output.append(o)
            return output
        return 'error'

    def filter_cube(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['shape'] == 'cube':
                    output.append(o)
            return output
        return 'error'

    def filter_cylinder(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['shape'] == 'cylinder':
                    output.append(o)
            return output
        return 'error'

    def filter_sphere(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['shape'] == 'sphere':
                    output.append(o)
            return output
        return 'error'

    def filter_large(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['size'] == 'large':
                    output.append(o)
            return output
        return 'error'

    def filter_small(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['size'] == 'small':
                    output.append(o)
            return output
        return 'error'

    def greater_than(self, integer1, integer2):
        if type(integer1) == int and type(integer2) == int:
            if integer1 > integer2:
                return 'yes'
            else:
                return 'no'
        return 'error'

    def less_than(self, integer1, integer2):
        if type(integer1) == int and type(integer2) == int:
            if integer1 < integer2:
                return 'yes'
            else:
                return 'no'
        return 'error'

    def intersect(self, scene1, scene2):
        if type(scene1) == list and type(scene2) == list:
            output = []
            for o in scene1:
                if o in scene2:
                    output.append(o)
            return output
        return 'error'

    def query_color(self, obj, _):
        if type(obj) == dict and 'color' in obj:
            return obj['color']
        return 'error'

    def query_material(self, obj, _):
        if type(obj) == dict and 'material' in obj:
            return obj['material']
        return 'error'

    def query_shape(self, obj, _):
        if type(obj) == dict and 'shape' in obj:
            return obj['shape']
        return 'error'

    def query_size(self, obj, _):
        if type(obj) == dict and 'size' in obj:
            return obj['size']
        return 'error'

    def relate_behind(self, obj, scene):
        if type(obj) == dict and 'position' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['position'][1] < obj['position'][1]:
                    output.append(o)
            return output
        return 'error'

    def relate_front(self, obj, scene):
        if type(obj) == dict and 'position' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['position'][1] > obj['position'][1]:
                    output.append(o)
            return output
        return 'error'

    def relate_left(self, obj, scene):
        if type(obj) == dict and 'position' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['position'][0] < obj['position'][0]:
                    output.append(o)
            return output
        return 'error'

    def relate_right(self, obj, scene):
        if type(obj) == dict and 'position' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['position'][0] > obj['position'][0]:
                    output.append(o)
            return output
        return 'error'

    def same_color(self, obj, scene):
        if type(obj) == dict and 'color' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['color'] == obj['color'] and o['id'] != obj['id']:
                    output.append(o)
            return output
        return 'error'

    def same_material(self, obj, scene):
        if type(obj) == dict and 'material' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['material'] == obj['material'] and o['id'] != obj['id']:
                    output.append(o)
            return output
        return 'error'

    def same_shape(self, obj, scene):
        if type(obj) == dict and 'shape' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['shape'] == obj['shape'] and o['id'] != obj['id']:
                    output.append(o)
            return output
        return 'error'

    def same_size(self, obj, scene):
        if type(obj) == dict and 'size' in obj and type(scene) == list:
            output = []
            for o in scene:
                if o['size'] == obj['size'] and o['id'] != obj['id']:
                    output.append(o)
            return output
        return 'error'

    def union(self, scene1, scene2):
        if type(scene1) == list and type(scene2) == list:
            output = list(scene2)
            for o in scene1:
                if o not in scene2:
                    output.append(o)
            return output
        return 'error'

    def unique(self, scene, _):
        if type(scene) == list and len(scene) > 0:
            return scene[0]
        return 'error'


if __name__ == '__main__':
    import json
    from tqdm import tqdm

    qf = r'D:\home\dataset\clevr\CLEVR_v1.0\questions\CLEVR_val_questions.json'
    sf = r'D:\home\dataset\clevr\CLEVR_v1.0\scenes\CLEVR_val_scenes.json'
    df = r"CLEVR_val_questions_with_ans.jsonl"

    ce = ClevrExecutor()
    qv = json.load(open(qf, 'r'))
    sv = json.load(open(sf, 'r'))

    for q in tqdm(qv['questions']):
        s = sv['scenes'][q['image_index']]
        ans = ce.run_question(q, s)
        assert ans[0] == q['answer']
        program2ans = {}
        for item in ans[1]:
            if isinstance(item['ans'], list):
                program2ans[item['idx']] = [_['object_index'] for _ in item['ans']]
            elif isinstance(item['ans'], dict) and 'object_index' in item['ans']:
                program2ans[item['idx']] = [item['ans']['object_index'], ]
            elif isinstance(item['ans'], (str, int)):
                program2ans[item['idx']] = item['ans']
            else:
                assert False, f"{item}"

        program = q['program']
        if 0 in program2ans:
            for idx, item in enumerate(program):
                item['ans'] = program2ans[idx]
        else:
            # hack: a bug in origin function label
            assert program[0]['function'] == 'scene'
            assert program[1]['function'] == 'scene'
            cnt1 = 0
            for _item in program:
                assert 0 not in _item['inputs']
                if 1 in _item['inputs']:
                    cnt1 += 1
                    if cnt1 == 2:
                        assert len(_item['inputs']) == 2
                        new_values = [v if v != 1 else 0 for v in _item['inputs']]
                        _item['inputs'] = new_values
            assert cnt1 == 2
            program2ans[0] = program2ans[1]
            for idx, item in enumerate(program):
                item['ans'] = program2ans[idx]

        assert str(program[-1]['ans']) == str(q['answer']), f"{q}"

    if str(df).endswith('.jsonl'):
        import jsonlines
        with jsonlines.open(df, mode='w') as writer:
            for item in qv['questions']:
                writer.write(item)
    elif str(df).endswith('json'):
        json.dump(qv, open(df, 'w'))
    else:
        assert False
