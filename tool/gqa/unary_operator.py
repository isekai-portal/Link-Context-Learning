from root import PHRASE_ST_PLACEHOLDER, PHRASE_ED_PLACEHOLDER, get_box_xyxy

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

    def chain_to_str(self, chain) -> (list, str):
        # select
        # filter - select
        # exist - filter - select
        # relate - filter - select
        # query - relate - filter - select
        return ""

    def chain_to_str_query(self, chain):
        assert chain[0] == len(self.operations) - 1
        assert self.operations[chain[0]]['operation'] == 'query'
        assert len(chain) > 1
        rid, ret = self.chain_to_str_with_obj(chain[1:])

        ref = self.scene['objects'][rid]
        ref_box = get_box_xyxy(ref)
        idx = self.get_boxes_idx(ref_box)
        argu = self.operations[chain[0]]['argument']
        query_str = f"Let's take a closer look at the {argu} of {PHRASE_ST_PLACEHOLDER} the object {PHRASE_ST_PLACEHOLDER}."

        return " ".join([ret, query_str]).strip()

    def chain_to_str_relate(self, chain):
        assert chain[0] == len(self.operations) - 1
        assert self.operations[chain[0]]['operation'] == 'relate'
        assert len(chain) > 1
        obj_id, ret = self.chain_to_str_with_obj(chain[1:])
        argu = self.operations[chain[0]]['argument']
        obj, relation, src, target = None, None, None, None

        query_str = f"."
        return " ".join([ret, query_str]).strip()

    def chain_to_str_with_obj(self, chain):
        return obj_id, ret
