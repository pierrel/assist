"""ImageInputGuardMiddleware — the text-only model must never receive an image block
(read_file on a .png asset would 500 the llama.cpp server otherwise)."""
from langchain_core.messages import AIMessage, HumanMessage

from assist.middleware.image_input_guard import (
    ImageInputGuardMiddleware,
    _has_image_block,
)


class _FakeReq:
    def __init__(self, messages):
        self.messages = messages

    def override(self, messages):
        return _FakeReq(messages)


def _img_msg():
    return HumanMessage(content=[
        {"type": "text", "text": "here's the wallpaper"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ])


def test_image_block_stripped_before_model_call():
    captured = {}

    def handler(req):
        captured["msgs"] = req.messages
        return AIMessage(content="ok")

    ImageInputGuardMiddleware().wrap_model_call(_FakeReq([_img_msg()]), handler)
    sent = captured["msgs"][0].content
    assert not _has_image_block(sent)                                  # no image reaches the model
    assert any(b.get("type") == "text" and "cannot view images" in b["text"] for b in sent)
    assert any(b.get("text") == "here's the wallpaper" for b in sent)  # other blocks preserved


def test_plain_text_is_passthrough_untouched():
    orig = [HumanMessage(content="plain text")]
    seen = {}

    def handler(req):
        seen["req"] = req
        return AIMessage(content="ok")

    ImageInputGuardMiddleware().wrap_model_call(_FakeReq(orig), handler)
    assert seen["req"].messages is orig   # no image → passed straight through, no copy/override
