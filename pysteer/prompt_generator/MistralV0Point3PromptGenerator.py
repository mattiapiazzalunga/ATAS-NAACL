from prompt_generator.BasePromptGenerator import BasePromptGenerator


class MistralV0Point3PromptGenerator(BasePromptGenerator):

    @staticmethod
    def _template_for_base_models(system: str, user: str) -> str:
        return f"<s>{system}\n\n{user}"

    @staticmethod
    def _template_for_instruct_models(who_you_are: str, system: str, user: str) -> str:
        return (
            "<s>[INST] "
            f"You are an expert in {who_you_are}. {system}\n\n{user}[/INST]"
        )
    # The generate_prompt() method is inherited from BasePromptGenerator.
