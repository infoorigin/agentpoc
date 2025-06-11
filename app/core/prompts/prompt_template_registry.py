import json
import os
from typing import Dict



class PromptTemplateRegistry:
    """
    Central registry for multilingual prompt templates by category (e.g., 'agent','tool' ..).
    Automatically loads built-in templates from `prompt_templates.json` in the same folder.
    """

    _registry: Dict[str, Dict[str, Dict[str, str]]] = {}
    _default_language = "en"
    _initialized = False

    @classmethod
    def _initialize(cls):
        if cls._initialized:
            return

        dir_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(dir_path, "prompt_templates.json")

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for category, templates_by_language in data.items():
                    cls.load_templates(category, templates_by_language, internal=True)

        cls._initialized = True

    @classmethod
    def load_templates(cls, category: str, templates_by_language: Dict[str, Dict[str, str]], internal: bool = False):
        if not internal:
            cls._initialize()
        cls._registry.setdefault(category, {}).update(templates_by_language)

    @classmethod
    def register_template(cls, category: str, lang_code: str, template_dict: Dict[str, str]):
        """
        Register or override templates for a specific category and language.

        Args:
            category (str): The template category (e.g., "agent", "tool").
            lang_code (str): The language code (e.g., "en", "fr", "it").
            template_dict (Dict[str, str]): A dictionary with keys like:
                - "system_prompt": the main prompt template
                - "goal_block": optional add-on for goal
                - "description": a short description of the persona

        Example:
            >>> PromptTemplateRegistry.register_template("agent", "it", {
            ...     "system_prompt": "Sei {role}. {background}{goal_block}",
            ...     "goal_block": "\\nIl tuo obiettivo personale è: {goal}",
            ...     "description": "Un agente persona che agisce come {role}."
            ... })
        """
        cls._initialize()
        cls._registry.setdefault(category, {})[lang_code] = template_dict


    @classmethod
    def get_templates(cls, category: str, lang_code: str) -> Dict[str, str]:
        cls._initialize()
        if category not in cls._registry:
            raise ValueError(f"No templates registered for category '{category}'.")

        return cls._registry[category].get(lang_code) or cls._registry[category].get(cls._default_language, {})
