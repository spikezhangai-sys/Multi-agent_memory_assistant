from driftscope.core.topic_tree import TopicTree


def test_load_default_topic_tree() -> None:
    tree = TopicTree.load_default()
    assert tree.has_topic("user.preference.food")
    assert tree.has_topic("user.profile.location")
    assert tree.has_topic("user.profile.education")
    assert tree.has_topic("user.preference.software")
    assert tree.has_topic("user.activity.errands")
    assert tree.has_topic("user.activity.cultural_visit")
    assert tree.has_topic("user.metric.performance")


def test_match_food_preference() -> None:
    tree = TopicTree.load_default()
    assert tree.match("我最近真的很喜欢吃日料") == "user.preference.food"


def test_match_education_fact() -> None:
    tree = TopicTree.load_default()
    assert tree.match("I graduated with a degree in Business Administration.") == "user.profile.education"


def test_match_software_preference() -> None:
    tree = TopicTree.load_default()
    assert tree.match("I'm trying to learn more about advanced settings in Adobe Premiere Pro.") == "user.preference.software"


def test_match_errand_fact() -> None:
    tree = TopicTree.load_default()
    assert tree.match("I still need to return some boots and pick up the new pair from Zara.") == "user.activity.errands"


def test_match_cultural_visit_fact() -> None:
    tree = TopicTree.load_default()
    assert tree.match("I visited the Museum of Modern Art for a guided tour.") == "user.activity.cultural_visit"


def test_match_performance_metric_fact() -> None:
    tree = TopicTree.load_default()
    assert tree.match("I'm hoping to beat my personal best time of 25:50 this time around.") == "user.metric.performance"


def test_match_returns_none_when_signal_is_too_ambiguous() -> None:
    tree = TopicTree.load_default()
    assert tree.match("I need help with food, software, workouts, and errands.") is None


def test_match_returns_none_when_no_signal() -> None:
    tree = TopicTree.load_default()
    assert tree.match("zxqv lorem ipsum") is None
