from typing import cast

from enums.GlueTaskEnum import GlueTaskEnum
from predefined_benchmarks.BaseBenchmarkPromptGenerator import BaseBenchmarkPromptGenerator
from utils.StringUtils import StringUtils


class GlueBenchmarkPromptGenerator(BaseBenchmarkPromptGenerator):

    @property
    def task(self) -> GlueTaskEnum:
        return cast(GlueTaskEnum, self._task)

    @task.setter
    def task(self, task: GlueTaskEnum):
        if task is None or not isinstance(task, GlueTaskEnum):
            raise ValueError("Task must not be None and must be an instance of GlueTaskEnum.")
        self._task = task

    def _get_who_you_are(self) -> str:
        expert_descriptions = {
            GlueTaskEnum.MNLI: "linguistic analysis, logical reasoning, and natural language inference",
            GlueTaskEnum.QQP: "semantic analysis, paraphrase detection, and comparative linguistic evaluation",
            GlueTaskEnum.QNLI: "contextual understanding, natural language inference, and semantic evaluation",
            GlueTaskEnum.SST_2: "sentiment analysis, emotional nuance, and linguistic evaluation",
            GlueTaskEnum.CoLA: "grammatical analysis, syntax evaluation, and linguistic acceptability",
            GlueTaskEnum.STS_B: "semantic similarity assessment, linguistic semantics, and comparative evaluation",
            GlueTaskEnum.MRPC: "paraphrase recognition, semantic equivalence, and linguistic reasoning",
            GlueTaskEnum.RTE: "logical inference, reasoning, and language understanding",
            GlueTaskEnum.WNLI: "Ambiguity resolution, pronoun disambiguation, and nuanced language understanding"
        }

        return expert_descriptions[self.task]

    def _get_system_message(self) -> str:
        system_messages = {
            GlueTaskEnum.MNLI: (
                "MNLI (Multi-Genre Natural Language Inference) is a task where you classify the relationship "
                "between a premise and a hypothesis. The possible categories are: 'entailment', 'contradiction', "
                "or 'neutral'. 'entailment' means the hypothesis follows from the premise, 'contradiction' means "
                "they conflict, and 'neutral' means neither clearly supported nor refuted.\n"
                "Given a premise and a hypothesis, identify which of these three categories applies.\n"
                "Answer only with 'entailment', 'contradiction', or 'neutral'."
            ),
            GlueTaskEnum.QQP: (
                "QQP (Quora Question Pairs) is a task where you determine whether two questions have the same meaning. "
                "The possible categories are: 'duplicate' or 'not_duplicate'. Consider the semantics of each question "
                "to decide if they convey the same intent.\n"
                "Given two questions, choose the correct category.\n"
                "Answer only with 'duplicate' or 'not_duplicate'."
            ),
            GlueTaskEnum.QNLI: (
                "QNLI (Question Natural Language Inference) is a task where you decide whether a given sentence entails "
                "the answer to a question. The possible categories are: 'entailment' or 'not_entailment'.\n"
                "Given a question and a sentence, determine if the sentence contains the answer.\n"
                "Answer only with 'entailment' or 'not_entailment'."
            ),
            GlueTaskEnum.SST_2: (
                "SST-2 (Stanford Sentiment Treebank) is a task where you classify the sentiment of a sentence. "
                "The possible categories are: 'positive' or 'negative'.\n"
                "Given a sentence, decide which sentiment applies.\n"
                "Answer only with 'positive' or 'negative'."
            ),
            GlueTaskEnum.CoLA: (
                "CoLA (Corpus of Linguistic Acceptability) is a task where you judge whether a sentence is grammatically "
                "acceptable in standard English. The possible categories are: 'acceptable' or 'unacceptable'.\n"
                "Given a sentence, decide which category applies.\n"
                "Answer only with 'acceptable' or 'unacceptable'."
            ),
            GlueTaskEnum.STS_B: (
                "STS-B (Semantic Textual Similarity Benchmark) is a task where you rate the semantic similarity between "
                "two sentences on a scale from 0 to 5. The possible scores are floating-point numbers between 0 and 5\n"
                "Given two sentences, choose the numeric similarity score.\n"
                "Answer only with a floating-point number from 0 to 5."
            ),
            GlueTaskEnum.MRPC: (
                "MRPC (Microsoft Research Paraphrase Corpus) is a task where you determine if two sentences are "
                "paraphrases (equivalent) of each other. The possible categories are: 'equivalent' or 'not_equivalent'.\n"
                "Given two sentences, decide which category applies.\n"
                "Answer only with 'equivalent' or 'not_equivalent'."
            ),
            GlueTaskEnum.RTE: (
                "RTE (Recognizing Textual Entailment) is a task where you decide if a hypothesis is entailed by a "
                "premise. The possible categories are: 'entailment' or 'not_entailment'.\n"
                "Given a premise and a hypothesis, choose the correct category.\n"
                "Answer only with 'entailment' or 'not_entailment'."
            ),
            GlueTaskEnum.WNLI: (
                "WNLI (Winograd NLI) is a task that tests whether a hypothesis is entailed by a premise, often requiring pronoun resolution "
                "to determine the correct referent. The possible categories are: "
                "'entailment' or 'not_entailment'.\n"
                "Given a premise and a hypothesis, choose the correct category.\n"
                "Answer only with 'entailment' or 'not_entailment'."
            ),
        }

        return system_messages[self.task]

    def _get_user_message(self, sentence1_or_premise_or_question: str,
                          sentence_2_or_hypothesis_or_answer: str = None) -> str:
        user_messages = {
            GlueTaskEnum.MNLI: (
                f"We have three categories: 'entailment', 'contradiction', or 'neutral'.\n"
                f"Premise: '{sentence1_or_premise_or_question}'\n"
                f"Hypothesis: '{sentence_2_or_hypothesis_or_answer}'\n"
                "Which category best describes their relationship?"
            ),
            GlueTaskEnum.QQP: (
                f"We have two categories: 'duplicate' or 'not_duplicate'.\n"
                f"Question 1: '{sentence1_or_premise_or_question}'\n"
                f"Question 2: '{sentence_2_or_hypothesis_or_answer}'\n"
                "Do these questions have the same meaning?"
            ),
            GlueTaskEnum.QNLI: (
                f"We have two categories: 'entailment' or 'not_entailment'.\n"
                f"Question: '{sentence1_or_premise_or_question}'\n"
                f"Sentence: '{sentence_2_or_hypothesis_or_answer}'\n"
                "Does the sentence entail the answer to the question?"
            ),
            GlueTaskEnum.SST_2: (
                f"We have two categories: 'positive' or 'negative'.\n"
                f"Sentence: '{sentence1_or_premise_or_question}'\n"
                "Which sentiment applies?"
            ),
            GlueTaskEnum.CoLA: (
                f"We have two categories: 'acceptable' or 'unacceptable'.\n"
                f"Sentence: '{sentence1_or_premise_or_question}'\n"
                "Is this sentence grammatically acceptable?"
            ),
            GlueTaskEnum.STS_B: (
                f"We have possible similarity scores from 0 to 5 (floating-point).\n"
                f"Sentence 1: '{sentence1_or_premise_or_question}'\n"
                f"Sentence 2: '{sentence_2_or_hypothesis_or_answer}'\n"
                "How similar in meaning are these sentences?"
            ),
            GlueTaskEnum.MRPC: (
                f"We have two categories: 'equivalent' or 'not_equivalent'.\n"
                f"Sentence 1: '{sentence1_or_premise_or_question}'\n"
                f"Sentence 2: '{sentence_2_or_hypothesis_or_answer}'\n"
                "Do these sentences convey the same meaning?"
            ),
            GlueTaskEnum.RTE: (
                f"We have two categories: 'entailment' or 'not_entailment'.\n"
                f"Premise: '{sentence1_or_premise_or_question}'\n"
                f"Hypothesis: '{sentence_2_or_hypothesis_or_answer}'\n"
                "Does the premise entail the hypothesis?"
            ),
            GlueTaskEnum.WNLI: (
                f"We have two categories: 'entailment' or 'not_entailment'.\n"
                f"Premise: '{sentence1_or_premise_or_question}'\n"
                f"Hypothesis: '{sentence_2_or_hypothesis_or_answer}'\n"
                "Does the premise entail the hypothesis?"
            ),
        }

        return user_messages[self.task]

    def generate_prompt_for_the_task(
            self,
            sentence1_or_premise_or_question: str,
            sentence_2_or_hypothesis_or_answer: str = None
    ) -> str:
        one_sentence_tasks = {GlueTaskEnum.SST_2, GlueTaskEnum.CoLA}
        two_sentence_tasks = {
            GlueTaskEnum.MNLI, GlueTaskEnum.QQP, GlueTaskEnum.QNLI,
            GlueTaskEnum.STS_B, GlueTaskEnum.MRPC, GlueTaskEnum.RTE,
            GlueTaskEnum.WNLI
        }

        sentence1_or_premise_or_question = StringUtils.trim_string(StringUtils.remove_spaces(sentence1_or_premise_or_question))

        if self.task in one_sentence_tasks and sentence_2_or_hypothesis_or_answer is not None:
            raise ValueError(f"The task {self.task.name} requires only one sentence.")
        if self.task in two_sentence_tasks and sentence_2_or_hypothesis_or_answer is None:
            raise ValueError(f"The task {self.task.name} requires two sentences.")

        if sentence_2_or_hypothesis_or_answer is not None:
            sentence_2_or_hypothesis_or_answer = StringUtils.trim_string(StringUtils.remove_spaces(sentence_2_or_hypothesis_or_answer))

        who_you_are = self._get_who_you_are()
        system = self._get_system_message()
        user = self._get_user_message(sentence1_or_premise_or_question, sentence_2_or_hypothesis_or_answer)

        return self._generate_prompt_based_on_the_model(system, user, who_you_are)
