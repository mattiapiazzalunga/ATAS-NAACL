from prompt_generator.BasePromptGenerator import BasePromptGenerator


class OLMo2PromptGenerator(BasePromptGenerator):

    @staticmethod
    def _template_for_base_models(system: str, user: str) -> str:
        return f"{system}\n\n{user}\n"

    @staticmethod
    def _template_for_instruct_models(who_you_are: str, system: str, user: str) -> str:
        system_parts = []
        if who_you_are:
            system_parts.append(f"You are an expert in {who_you_are}.")
        if system:
            system_parts.append(system)
        system_content = " ".join(system_parts)
        return (
            "<|endoftext|><|system|>\n"
            f"{system_content}\n"
            "<|user|>\n"
            f"{user}\n"
            "<|assistant|>\n"
        )

    # The generate_prompt() method is inherited from BasePromptGenerator.
