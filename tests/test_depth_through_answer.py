"""scripts/depth_through_answer.py on the CPU: the positions it reads in each model's answer-time sequence (the token
strings below are real tokenizations of each model), the rule for the token right before the answer, the drawing order
and the shards, and the two-of-three seed rule."""

from vlens.results import load_script

s = load_script("depth_through_answer")

TEXT = "What do people drink that comes from the farm animal that says moo?"
ASKED = TEXT + " Answer with one word."
Q = ["What", " do", " people", " drink", " that", " comes", " from", " the", " farm", " animal", " that", " says", " moo",
     "?"]
INSTR = [" Answer", " with", " one", " word", "."]
ANSWERS = ["milk"]


def check(strs, content, reply, want_pre, want_new):
    q_end, instr = s.question_spans(strs, TEXT, ASKED)
    assert strs[q_end] == "?"
    pre, ans = s.answer_spot(strs, content, reply, ANSWERS)
    assert strs[pre] == want_pre
    new = s.new_positions(strs, q_end, instr, pre, ans)
    assert [(x["token"], x["group"][0]) for x in new] == want_new
    return q_end, pre, ans


def test_bagel():
    """read_reply's two messages: <|im_start|> question <|im_end|>, then <|im_start|> reply <|im_end|>."""
    first = ["<|im_start|>"] + Q + INSTR + ["<|im_end|>"]
    rep = ["<|im_start|>", "Milk", "<|im_end|>"]
    content = [False] * len(first) + [False, True, False]
    check(first + rep, content, "Milk", "<|im_start|>",
          [(t, "i") for t in INSTR] + [("<|im_end|>", "c"), ("Milk", "a")])


def test_show_o2():
    head = ["<|im_start|>", "system", "\n", "You", " are", " a", " helpful", " assistant", ".", "<|im_end|>", "\n",
            "<|im_start|>", "user", "\n"]
    tail = ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"]
    strs = head + Q + INSTR + tail + ["Milk"]
    content = [False] * (len(strs) - 1) + [True]
    check(strs, content, "Milk", "\n", [(t, "i") for t in INSTR] + [(t, "c") for t in tail[:-1]] + [("Milk", "a")])


def test_sensenova():
    """Its real tokenization (no system message; "Milk" is two tokens, so the answer token is "M")."""
    tail = ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n"]
    strs = ["<|im_start|>", "user", "\n"] + Q + INSTR + tail + ["M", "ilk"]
    content = [False] * (len(strs) - 2) + [True, True]
    check(strs, content, "Milk", "\n\n", [(t, "i") for t in INSTR] + [(t, "c") for t in tail[:-1]] + [("M", "a")])


SYS = ["<|startoftext|>", "You", " are", " a", " helpful", " assistant", ".", " Answer", " every", " question", " with",
       " exactly", " one", " word", ".", " Never", " explain", " your", " answer", ".", "\n\n", "User", ":", " "]
HY_TAIL = ["\n\n", "Assistant", ":", " "]


def test_hunyuan_plain_reply():
    """The prompt ends with "Assistant: " (its last token a space, the pre-answer token); the reply starts with the
    answer. The system prompt's "Answer every question with exactly one word." is not the instruction."""
    strs = SYS + Q + INSTR + HY_TAIL + ["Milk"]
    content = [False] * (len(strs) - 1) + [True]
    check(strs, content, "Milk", " ", [(t, "i") for t in INSTR] + [(t, "c") for t in HY_TAIL[:-1]] + [("Milk", "a")])


def test_hunyuan_bold_reply():
    """cow_milk: the reply opens with bold markup, so the token before the answer is the generated " **" and the
    prompt's last space becomes a new chat position."""
    strs = SYS + Q + INSTR + HY_TAIL + [" **", "Milk"]
    content = [False] * (len(strs) - 2) + [True, True]
    check(strs, content, "**Milk**", " **", [(t, "i") for t in INSTR] + [(t, "c") for t in HY_TAIL] + [("Milk", "a")])


def test_answer_spot_on_preamble():
    """A reply with words before the answer: the token before the answer is the reply's own previous token."""
    strs = ["<|im_start|>"] + Q + INSTR + ["<|im_end|>", "<|im_start|>", "It", "'s", " milk", "."]
    content = [False] * (len(strs) - 4) + [True] * 4
    q_end, pre, ans = check(strs, content, "It's milk.", "'s",
                            [(t, "i") for t in INSTR] + [("<|im_end|>", "c"), ("<|im_start|>", "c"), ("It", "c"),
                                                         (" milk", "a")])


def test_order_and_shards():
    plan = dict(questions=[dict(item=i, positions=[dict(p=p, group=g, token="") for p, g in
                                                   enumerate(["instruction"] * 5 + ["chat"] * (i % 3 + 1) + ["answer"])])
                           for i in (0, 2, 5, 7)])
    u = s.units_of(plan)
    ranks = [s.RANK[x["group"]] for _, x in u]
    assert ranks == sorted(ranks)
    for of in (1, 3, 7, 24):
        got = [k for shard in range(of) for k in range(len(u))[s.shard_slice(len(u), shard, of)]]
        assert got == list(range(len(u)))


def test_decide_two_of_three():
    assert s.decide({3: True, 4: True}) is True
    assert s.decide({3: False, 4: False}) is False
    assert s.decide({3: True, 4: False}) is None
    assert s.decide({3: True, 4: False, 5: True}) is True
    assert s.decide({3: True, 4: None}) is None
