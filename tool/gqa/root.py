PHRASE_ST_PLACEHOLDER = '<ph_st>'
PHRASE_ED_PLACEHOLDER = '<ph_ed>'
CANT_INFER_RESULT = "CAN'T INFER RESULT"


def get_box_xyxy(obj):
    x: int
    y: int
    w: int
    h: int
    x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
    return x, y, x + w, y + h
