"""Execution modes: how a prompt is asked, and how the answers come back.

These cover the parts of the specification that change what a *number means*
rather than how it is computed -- a run labelled ``BOV`` must actually have
asked one question per hypothesis, a run labelled ``self-consistency`` must
actually have voted, and a dataset whose task requires several selections must
never be offered a single-choice mode.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from abductionbench.adapters._interactive import EvidenceStore, parse_action
from abductionbench.adapters._prompting import PromptParts, build_messages
from abductionbench.core.adapter import AdapterContext, DatasetAdapter
from abductionbench.core.errors import ConfigError
from abductionbench.core.modes import TaskModes, template_mode_of
from abductionbench.core.types import (
    ChatMessage,
    ModelResponse,
    ResponseStatus,
    SampleScore,
    SampleSpec,
)


class _Selection(DatasetAdapter):
    """A minimal selection dataset, used to exercise the mode machinery."""

    system_prompt = "You pick the best explanation."
    objective_metrics = True
    selection_cardinality = "flexible"

    def build_samples(self):
        return [
            SampleSpec(
                sample_id="s1",
                fields={
                    "observation": "the lawn is wet",
                    "options": ["it rained", "the sprinkler ran", "a pipe burst"],
                    "option_labels": ["A", "B", "C"],
                },
                reference={"gold_labels": ["A", "B"]},
                task_kind="selection",
            )
        ]

    def build_messages(self, sample):
        return build_messages(
            PromptParts(
                system=self.system_prompt_for(sample),
                observation=str(sample.fields["observation"]),
                options=list(sample.fields["options"]),
                option_labels=list(sample.fields["option_labels"]),
            ),
            self.context.modes,
        )

    def score(self, sample, response, *, output_contract=None):
        picked = {
            token.strip().upper()
            for token in response.text.replace("Answer:", "").split(",")
            if token.strip()
        }
        gold = set(sample.reference["gold_labels"])
        return SampleScore(
            metrics={"set_f1": 1.0 if picked == gold else 0.0},
            prediction=",".join(sorted(picked)),
        )

    def aggregate(self, scores):
        return {"set_f1": sum(s.metrics["set_f1"] for s in scores) / max(1, len(scores))}

    def documentation(self):  # pragma: no cover - not exercised here
        raise NotImplementedError


def _adapter(modes: TaskModes, cls=_Selection):
    return cls(AdapterContext(dataset_id="t", data_dir=Path("/tmp"), modes=modes))


def _response(text: str) -> ModelResponse:
    return ModelResponse(sample_id="s", model_id="m", status=ResponseStatus.OK, content=text)


# --------------------------------------------------------------------------- #
# the mode vocabulary and the identity it composes
# --------------------------------------------------------------------------- #


def test_template_mode_is_the_four_axes():
    assert template_mode_of("cot", "BOV", "selection", "interactive") == (
        "cot|BOV|selection|interactive"
    )
    # A generation task has no selection axis, and says so rather than leaving a
    # blank that cannot be filtered on.
    assert template_mode_of("io", None, "generation", "static") == "io|n/a|generation|static"


def test_unknown_modes_are_rejected_at_construction():
    for bad in ({"prompt_mode": "reasoning"}, {"selection_mode": "multi"},
                {"data_delivery_mode": "streaming"}):
        with pytest.raises(ConfigError):
            TaskModes(**bad)
    with pytest.raises(ConfigError, match="majority"):
        TaskModes(prompt_mode="self-consistency", self_consistency_n=1)


def test_slug_is_filesystem_safe_and_distinct_per_combination():
    slugs = {
        TaskModes(prompt_mode=p, selection_mode=s).slug
        for p in ("io", "cot")
        for s in (None, "SCS", "MCS", "BOV")
    }
    assert len(slugs) == 8
    assert all("/" not in slug and "|" not in slug for slug in slugs)


# --------------------------------------------------------------------------- #
# which modes a dataset admits
# --------------------------------------------------------------------------- #


def test_reasoning_modes_are_only_for_objectively_scored_datasets():
    class Subjective(_Selection):
        objective_metrics = False

    assert Subjective.supports_modes(TaskModes(prompt_mode="io")) is None
    problem = Subjective.supports_modes(TaskModes(prompt_mode="cot"))
    assert problem and "objectively verifiable" in problem


def test_a_multi_answer_benchmark_is_never_offered_single_choice():
    class MultiOnly(_Selection):
        selection_cardinality = "multi"

    assert MultiOnly.selection_modes_offered() == ["MCS", "BOV"]
    problem = MultiOnly.supports_modes(TaskModes(selection_mode="SCS"))
    assert problem and "SCS is not offered" in problem
    assert MultiOnly.supports_modes(TaskModes(selection_mode="MCS")) is None


def test_a_single_answer_benchmark_is_never_offered_multi_choice():
    class SingleOnly(_Selection):
        selection_cardinality = "single"

    assert SingleOnly.selection_modes_offered() == ["SCS", "BOV"]
    assert SingleOnly.supports_modes(TaskModes(selection_mode="MCS"))


def test_a_flexible_benchmark_offers_all_three():
    assert _Selection.selection_modes_offered() == ["SCS", "MCS", "BOV"]


def test_a_generation_dataset_has_no_selection_mode():
    class Generative(_Selection):
        selection_cardinality = None

    assert Generative.selection_modes_offered() == []
    assert Generative.supports_modes(TaskModes(selection_mode="SCS"))


# --------------------------------------------------------------------------- #
# prompt modes
# --------------------------------------------------------------------------- #


def test_io_asks_for_the_answer_and_cot_asks_for_reasoning_first():
    direct = _adapter(TaskModes(prompt_mode="io", selection_mode="SCS"))
    reasoned = _adapter(TaskModes(prompt_mode="cot", selection_mode="SCS"))
    sample = direct.build_samples()[0]

    io_text = direct.build_messages(sample)[0][-1].content
    cot_text = reasoned.build_messages(sample)[0][-1].content
    assert "Do not explain your reasoning" in io_text
    assert "step by step" in cot_text
    # Same evidence, same answer contract: only the elicitation differs.
    for text in (io_text, cot_text):
        assert "the lawn is wet" in text
        assert "Answer:" in text


def test_self_consistency_asks_the_same_question_as_cot():
    """Otherwise a vote would confound the reasoning mode with the prompt."""
    reasoned = _adapter(TaskModes(prompt_mode="cot"))
    voting = _adapter(TaskModes(prompt_mode="self-consistency"))
    sample = reasoned.build_samples()[0]
    assert (
        reasoned.build_messages(sample)[0][-1].content
        == voting.build_messages(sample)[0][-1].content
    )


def test_the_system_prompt_comes_from_the_dataset():
    adapter = _adapter(TaskModes())
    messages, _ = adapter.build_messages(adapter.build_samples()[0])
    assert messages[0].role == "system"
    assert messages[0].content == "You pick the best explanation."


def test_no_universal_system_prompt_exists_in_the_core():
    """Item 4: the core must not impose one on every dataset."""
    import abductionbench.core as core_pkg

    root = Path(core_pkg.__file__).parent
    offenders = [
        path.name
        for path in root.glob("*.py")
        if "You are an expert" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


# --------------------------------------------------------------------------- #
# selection modes
# --------------------------------------------------------------------------- #


def test_scs_asks_for_one_and_mcs_asks_for_several():
    single = _adapter(TaskModes(selection_mode="SCS"))
    multi = _adapter(TaskModes(selection_mode="MCS"))
    sample = single.build_samples()[0]

    single_text, single_contract = single.build_messages(sample)
    multi_text, multi_contract = multi.build_messages(sample)
    assert "exactly one" in single_text[-1].content
    assert single_contract["style"] == "single_label"
    assert "one or several" in multi_text[-1].content
    assert multi_contract["style"] == "multi_label"
    # Both still show the same candidate list.
    for messages in (single_text, multi_text):
        assert "the sprinkler ran" in messages[-1].content


def test_bov_asks_one_question_per_hypothesis():
    adapter = _adapter(TaskModes(selection_mode="BOV"))
    derived = adapter.expand_for_modes(adapter.build_samples())
    assert len(derived) == 3
    assert [s.group_id for s in derived] == ["s1", "s1", "s1"]
    assert len({s.sample_id for s in derived}) == 3

    texts = [adapter.build_messages(s)[0][-1].content for s in derived]
    # Each request shows exactly one hypothesis, and asks a yes/no question.
    for text, hypothesis in zip(texts, ["it rained", "the sprinkler ran", "a pipe burst"],
                                strict=True):
        assert hypothesis in text
        assert "Answer: YES" in text
        others = {"it rained", "the sprinkler ran", "a pipe burst"} - {hypothesis}
        assert not any(other in text for other in others)


def test_bov_builds_the_selected_set_from_the_yeses():
    adapter = _adapter(TaskModes(selection_mode="BOV"))
    derived = adapter.expand_for_modes(adapter.build_samples())
    members = [
        (derived[0], _response("Answer: YES"), SampleScore()),
        (derived[1], _response("Answer: YES"), SampleScore()),
        (derived[2], _response("Answer: NO"), SampleScore()),
    ]
    reduced = adapter.reduce_group(members)
    assert reduced is not None
    assert reduced.prediction == "A,B"          # the gold set
    assert reduced.metrics["set_f1"] == 1.0
    assert reduced.details["bov_selected"] == ["A", "B"]
    assert reduced.metrics["bov_yes_rate"] == pytest.approx(2 / 3)


def test_only_a_direct_yes_selects_a_hypothesis():
    adapter = _adapter(TaskModes(selection_mode="BOV"))
    derived = adapter.expand_for_modes(adapter.build_samples())
    hedged = [
        (derived[0], _response("Answer: YES"), SampleScore()),
        (derived[1], _response("It could plausibly be this one."), SampleScore()),
        (derived[2], _response("Answer: NO, but it is possible"), SampleScore()),
    ]
    reduced = adapter.reduce_group(hedged)
    assert reduced.details["bov_selected"] == ["A"]


# --------------------------------------------------------------------------- #
# self-consistency
# --------------------------------------------------------------------------- #


def test_self_consistency_takes_the_majority_answer():
    adapter = _adapter(TaskModes(prompt_mode="self-consistency", self_consistency_n=5))
    derived = adapter.expand_for_modes(adapter.build_samples())
    assert len(derived) == 5
    assert {s.group_id for s in derived} == {"s1"}
    # Same prompt every time -- only the sampling differs.
    assert len({adapter.build_messages(s)[0][-1].content for s in derived}) == 1

    votes = ["A,B", "A,B", "C", "A,B", "A"]
    members = [
        (sample, _response(f"Answer: {vote}"), adapter.score(sample, _response(f"Answer: {vote}")))
        for sample, vote in zip(derived, votes, strict=True)
    ]
    reduced = adapter.reduce_group(members)
    assert reduced.prediction == "A,B"
    assert reduced.metrics["set_f1"] == 1.0
    assert reduced.metrics["self_consistency_agreement"] == pytest.approx(3 / 5)


def test_a_vote_nobody_could_parse_keeps_the_failure():
    adapter = _adapter(TaskModes(prompt_mode="self-consistency", self_consistency_n=3))
    derived = adapter.expand_for_modes(adapter.build_samples())
    unparsed = SampleScore(metrics={"set_f1": 0.0}, parse_ok=False)
    members = [(s, _response(""), unparsed) for s in derived]
    reduced = adapter.reduce_group(members)
    assert reduced is not None and reduced.parse_ok is False


def test_modes_compose_bov_inside_a_vote():
    adapter = _adapter(TaskModes(prompt_mode="self-consistency", self_consistency_n=2,
                                 selection_mode="BOV"))
    derived = adapter.expand_for_modes(adapter.build_samples())
    # Three hypotheses, each asked twice.
    assert len(derived) == 6
    assert {s.group_id for s in derived} == {"s1"}


# --------------------------------------------------------------------------- #
# interactive delivery
# --------------------------------------------------------------------------- #


def test_action_parsing_accepts_both_published_field_namings():
    actions = ("history", "askquestion", "diagnosis_final")
    viva = parse_action('{"action": "history", "query": "how long?"}', actions=actions)
    assert (viva.action, viva.query) == ("history", "how long?")

    evo = parse_action(
        '{"action_type": "AskQuestion", "action_text": "how long?"}', actions=actions
    )
    assert (evo.action, evo.query) == ("askquestion", "how long?")

    fenced = parse_action('```json\n{"action": "history", "query": "x"}\n```', actions=actions)
    assert fenced.action == "history"

    prose = parse_action("I have no idea what to do.", actions=actions)
    assert prose.parsed is False and prose.action == ""


def test_evidence_is_disclosed_only_when_it_is_asked_for():
    """The environment must answer the question, not hand over the case."""
    store = EvidenceStore(
        categories={
            "ask": {
                "Do you have a cough?": "yes",
                "Do you have diabetes?": "yes",
                "Do you have chest pain?": "no",
            }
        }
    )
    assert "cough" in store.reveal("ask", "do you have a cough?")
    # A question about something the record does not cover discloses nothing --
    # in particular not the answers to the questions that were not asked.
    assert store.reveal("ask", "have you been to west africa recently?") == ""
    assert store.reveal("ask", "any headaches?") == ""


def test_evidence_matching_survives_word_forms():
    store = EvidenceStore(
        categories={"test": {"0": "Histological findings: mature adipose tissue."}}
    )
    assert store.reveal("test", "biopsy and histopathology")


def test_a_default_episode_is_a_single_turn():
    """A dataset that does not implement an environment behaves as before."""
    adapter = _adapter(TaskModes())
    sample = adapter.build_samples()[0]
    messages, state = adapter.interactive_start(sample)
    assert messages == list(adapter.build_messages(sample)[0])
    assert adapter.interactive_step(sample, state, "Answer: A") is None


def test_derived_samples_keep_a_handle_on_the_item_they_came_from():
    adapter = _adapter(TaskModes(selection_mode="BOV"))
    derived = adapter.expand_for_modes(adapter.build_samples())
    parent = derived[0].metadata["_parent_sample"]
    assert parent.sample_id == "s1"
    # Underscore-prefixed, so it never reaches the serialized record.
    assert all(key.startswith("_") or key in ("bov_label", "bov_index", "bov_hypothesis")
               for key in derived[0].metadata)


def test_chat_message_roles_are_well_formed():
    adapter = _adapter(TaskModes(prompt_mode="cot", selection_mode="MCS"))
    messages, _ = adapter.build_messages(adapter.build_samples()[0])
    assert [m.role for m in messages] == ["system", "user"]
    assert all(isinstance(m, ChatMessage) and m.content.strip() for m in messages)


# --------------------------------------------------------------------------- #
# item 15: an introduced mode has to be justified, in the record
# --------------------------------------------------------------------------- #


def test_a_mode_the_table_lists_is_not_reported_as_introduced():
    class Listed(_Selection):
        hypothesis_modes = ("generation", "selection")
        table_hypothesis_mode = "Generation / Selection (separate tasks)"

    assert Listed.introduced_hypothesis_mode("generation") is None
    assert Listed.introduced_hypothesis_mode("selection") is None


def test_an_extra_mode_must_carry_the_benchmarks_own_formulation():
    class Extra(_Selection):
        hypothesis_modes = ("generation", "selection")
        table_hypothesis_mode = "Generation"
        hypothesis_mode_justification = "The release ships four answer options per case."

    assert Extra.introduced_hypothesis_mode("generation") is None
    assert Extra.introduced_hypothesis_mode("selection") == (
        "The release ships four answer options per case."
    )


def test_an_unjustified_extra_mode_says_so_rather_than_passing_silently():
    class Sloppy(_Selection):
        hypothesis_modes = ("generation", "selection")
        table_hypothesis_mode = "Generation"

    assert "bug in the adapter" in Sloppy.introduced_hypothesis_mode("selection")


def test_every_shipped_adapter_justifies_the_modes_it_adds():
    """No adapter may quietly invent a task the dataset table does not list."""
    import importlib
    import inspect
    import pkgutil

    import abductionbench.adapters as adapters
    from abductionbench.core.adapter import DatasetAdapter

    unjustified = []
    for module_info in pkgutil.iter_modules(adapters.__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"abductionbench.adapters.{module_info.name}")
        for _name, obj in vars(module).items():
            if not (inspect.isclass(obj) and issubclass(obj, DatasetAdapter)
                    and obj.__module__ == module.__name__):
                continue
            for mode in obj.hypothesis_modes:
                reason = obj.introduced_hypothesis_mode(mode)
                if reason and "bug in the adapter" in reason:
                    unjustified.append(f"{module_info.name}:{mode}")
    assert unjustified == []


# --------------------------------------------------------------------------- #
# a replacement item is a whole item, not its first request
# --------------------------------------------------------------------------- #


class _Replaceable(_Selection):
    """A selection dataset with a spare item to replace an oversize one."""

    def build_samples(self):
        return [self._item("s1", "the lawn is wet")]

    def replacement_samples(self, count, exclude):
        spare = self._item("s2", "the pavement is dry")
        return self.expand_for_modes([spare])

    @staticmethod
    def _item(sample_id, observation):
        return SampleSpec(
            sample_id=sample_id,
            fields={
                "observation": observation,
                "options": ["it rained", "the sprinkler ran", "a pipe burst"],
                "option_labels": ["A", "B", "C"],
            },
            reference={"gold_labels": ["A", "B"]},
            task_kind="selection",
        )


def test_a_bov_replacement_brings_all_of_its_hypotheses():
    """Taking only the first request would rebuild the item from a fragment."""
    adapter = _adapter(TaskModes(selection_mode="BOV"), cls=_Replaceable)
    replacements = adapter.replacement_samples(1, set())
    # One replacement item, three hypotheses, one group.
    assert len(replacements) == 3
    assert len({s.group_id for s in replacements}) == 1
    assert [s.metadata["bov_label"] for s in replacements] == ["A", "B", "C"]


def test_a_self_consistency_replacement_brings_all_of_its_votes():
    adapter = _adapter(
        TaskModes(prompt_mode="self-consistency", self_consistency_n=4), cls=_Replaceable
    )
    replacements = adapter.replacement_samples(1, set())
    assert len(replacements) == 4
    assert len({s.group_id for s in replacements}) == 1


# --------------------------------------------------------------------------- #
# repeats: the same record, asked several times, each scored on its own
# --------------------------------------------------------------------------- #


def test_repeats_ask_each_record_several_times_without_folding_them():
    adapter = _adapter(TaskModes(repeats=5))
    derived = adapter.expand_for_modes(adapter.build_samples())

    assert len(derived) == 5
    assert len({s.sample_id for s in derived}) == 5
    # No group_id: a repeat is an independent observation, not a piece of one.
    assert all(s.group_id is None for s in derived)
    assert {s.metadata["repeat_index"] for s in derived} == {0, 1, 2, 3, 4}
    assert {s.metadata["repeat_of"] for s in derived} == {"s1"}
    # Same question every time.
    assert len({adapter.build_messages(s)[0][-1].content for s in derived}) == 1
    # Nothing to reduce -- that is what separates repeats from a vote.
    assert TaskModes(repeats=5).needs_group_reduction is False


def test_repeats_are_drawn_warm_and_a_single_run_is_not():
    assert TaskModes(repeats=5, repeat_temperature=0.7).sampling_temperature == 0.7
    assert TaskModes(repeats=1).sampling_temperature is None
    # A vote still uses its own temperature even if repeats are also on.
    voting = TaskModes(prompt_mode="self-consistency", self_consistency_temperature=0.9, repeats=3)
    assert voting.sampling_temperature == 0.9


def test_repeats_compose_with_bov_and_with_a_vote():
    adapter = _adapter(TaskModes(repeats=3, selection_mode="BOV"))
    derived = adapter.expand_for_modes(adapter.build_samples())
    # 3 repeats x 3 hypotheses, grouped per repeat so each repeat rebuilds its
    # own selected set.
    assert len(derived) == 9
    assert len({s.group_id for s in derived}) == 3
    assert all(s.metadata["repeat_of"] == "s1" for s in derived)

    voting = _adapter(
        TaskModes(repeats=2, prompt_mode="self-consistency", self_consistency_n=4)
    )
    votes = voting.expand_for_modes(voting.build_samples())
    assert len(votes) == 8               # 2 repeats x 4 votes
    assert len({s.group_id for s in votes}) == 2


def test_the_record_behind_a_request_survives_every_expansion():
    from abductionbench.core.adapter import evaluation_item_id

    adapter = _adapter(TaskModes(repeats=2, selection_mode="BOV"))
    for sample in adapter.expand_for_modes(adapter.build_samples()):
        assert evaluation_item_id(sample) == "s1"
