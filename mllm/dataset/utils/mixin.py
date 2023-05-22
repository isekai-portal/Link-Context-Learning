import json
import random


class QuestionTemplateMixin:
    def __init__(self, *args, template_string=None, template_file=None, max_dynamic_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.template_string = template_string
        self.template_file = template_file
        self.max_dynamic_size = max_dynamic_size
        if template_string is None and template_file is None:
            raise ValueError("assign either template_string or template_file")
        if template_string is not None and template_file is not None:
            raise ValueError(f"assign both template_string and template_file:\nstring:{template_string}\nfile:{template_file}")
        if template_string is not None:
            self.templates = [self.template_string]
        else:
            assert template_file is not None
            self.templates = json.load(open(template_file, 'r', encoding='utf8'))
        if self.max_dynamic_size is not None:
            self.templates = self.templates[: self.max_dynamic_size]

    def get_template(self):
        return random.choice(self.templates)

    def template_nums(self):
        return len(self.templates)
