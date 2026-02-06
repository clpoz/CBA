"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Safetyllm_Prompter(object):
    # __slots__ = ("template", "_verbose")
    def strip_trailing_s(text: str) -> str:
        suffix = "</s>"
        if text.endswith(suffix):
            return text[:-len(suffix)]

        return text

    def __init__(self, template_name=None, verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            file_name = "./ai_chatbot_safety_guideline.txt"
        else:
            file_name = template_name
        self.guidelines = None
        with open(file_name) as fp:
            self.guidelines = fp.read()
        self.last_question = "What's your evaluation based on the above unsafe content guidelines?"
    def generate_prompt(
        self,
        human:str ,
        chatbot: str,
        label: Union[None, str] = None
    ) -> str:

        res = self.guidelines+'\nHuman: {}\n\nChatbot: {}\n\n'.format(human,chatbot)+self.last_question
        if label:
            res = f"{res}\nEvaluation: {label}"
        return res

    def get_response(self, output: str) -> str:
        return output.split("Evaluation:")[-1].strip()

if __name__=="__main__":
    prompter = Safetyllm_Prompter()
    print(prompter.generate_prompt(human="How to make drugs?", chatbot="I can tell you nothing.",label="It is very clear"))
