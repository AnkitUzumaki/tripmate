from evals.simulation import Conversation, score_conversation


def test_goal_phrases_found_across_turns_score_complete():
    conversation = Conversation(
        id="c1", turns=["going to Tokyo in December", "what should I pack?"],
        goal_phrases=["layer", "cold"],
    )
    result = score_conversation(conversation,
                                ["Great, Tokyo in December.",
                                 "Pack warm layers; it is cold."])
    assert result.goal_completion == 1.0


def test_partially_met_goals_score_between_zero_and_one():
    conversation = Conversation(id="c1", turns=["a", "b"],
                                goal_phrases=["layer", "umbrella"])
    result = score_conversation(conversation, ["ok", "pack layers"])
    assert result.goal_completion == 0.5


def test_context_is_retained_when_a_later_answer_names_the_destination():
    conversation = Conversation(
        id="c1", turns=["I'm going to Bangkok", "what should I pack?"],
        goal_phrases=[], context_phrases=["bangkok"],
    )
    result = score_conversation(conversation,
                                ["Nice choice.", "For Bangkok, pack light fabrics."])
    assert result.context_retained is True


def test_context_is_not_retained_when_later_answers_drop_the_destination():
    conversation = Conversation(
        id="c1", turns=["I'm going to Bangkok", "what should I pack?"],
        goal_phrases=[], context_phrases=["bangkok"],
    )
    result = score_conversation(conversation, ["Nice choice.", "Pack light fabrics."])
    assert result.context_retained is False


def test_a_conversation_without_goals_scores_one():
    conversation = Conversation(id="c1", turns=["hi"], goal_phrases=[])
    assert score_conversation(conversation, ["hello"]).goal_completion == 1.0
