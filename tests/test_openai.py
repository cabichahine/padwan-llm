import json
from contextlib import aclosing, nullcontext
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock

import pytest
from niquests.exceptions import HTTPError

from padwan_ai.errors import LLMError, QuotaExceededError, TooManyRequestsError
from padwan_ai.openai.batch import BatchJob, BatchResult
from padwan_ai.openai.client import (
    OPENAI_MODELS,
    OpenAIChatStream,
    OpenAIClient,
    _check_resp,
    _extract_text_payload,
    _extract_thought_payload,
)
from padwan_ai.openai.types import CreateChatCompletionStreamResponse


def test_openai_model_literal_includes_stable_drift_addition() -> None:
    assert "gpt-6.1-sol" in OPENAI_MODELS


@pytest.mark.parametrize(
    "status, json_data, headers, ctx",
    [
        pytest.param(200, {"ok": True}, None, nullcontext(), id="success"),
        pytest.param(
            429,
            {},
            {"retry-after": "30"},
            pytest.raises(TooManyRequestsError),
            id="429-retry",
        ),
        pytest.param(
            429, {}, None, pytest.raises(TooManyRequestsError), id="429-default-60"
        ),
        pytest.param(
            402,
            {"error": "quota"},
            None,
            pytest.raises(QuotaExceededError),
            id="402-quota",
        ),
        pytest.param(500, {"error": "server"}, None, pytest.raises(LLMError), id="500"),
    ],
)
def test_check_resp(status, json_data, headers, ctx, make_resp):
    resp = make_resp(status, json_data, headers)
    with ctx:
        result = _check_resp(resp)
        assert result == json_data


class TestOpenAIChatStreamExtraction:
    def setup_method(self):
        client = MagicMock(spec=OpenAIClient)
        self.stream = OpenAIChatStream(client, [])

    @pytest.mark.parametrize(
        "chunk, expected",
        [
            pytest.param(
                {"choices": [{"delta": {"content": "hello"}}]}, "hello", id="text"
            ),
            pytest.param({"choices": [{"delta": {}}]}, None, id="empty-delta"),
            pytest.param({}, None, id="empty"),
            pytest.param(
                {
                    "choices": [
                        {
                            "delta": {
                                "content": [
                                    {"type": "text", "text": "Hello "},
                                    {"type": "text", "text": "world"},
                                ]
                            }
                        }
                    ]
                },
                "Hello world",
                id="mistral-text-chunks",
            ),
            pytest.param(
                {
                    "choices": [
                        {
                            "delta": {
                                "content": [
                                    {
                                        "type": "thinking",
                                        "thinking": [
                                            {"type": "text", "text": "Thinking..."}
                                        ],
                                    }
                                ]
                            }
                        }
                    ]
                },
                None,
                id="mistral-thinking-only",
            ),
        ],
    )
    def test_extract_text(
        self, chunk: CreateChatCompletionStreamResponse, expected: str | None
    ):
        assert self.stream._extract_text(chunk) == expected

    @pytest.mark.parametrize(
        "chunk, expected",
        [
            pytest.param(
                {
                    "usage": {
                        "total_tokens": 100,
                        "prompt_tokens": 80,
                        "completion_tokens": 20,
                    }
                },
                {"total": 100, "input": 80, "output": 20},
                id="basic",
            ),
            pytest.param(
                {
                    "usage": {
                        "total_tokens": 100,
                        "prompt_tokens": 80,
                        "completion_tokens": 20,
                        "prompt_tokens_details": {"cached_tokens": 10},
                    }
                },
                {"total": 100, "input": 80, "output": 20, "cached": 10},
                id="cached",
            ),
            pytest.param({}, None, id="no-usage"),
        ],
    )
    def test_extract_usage(
        self, chunk: CreateChatCompletionStreamResponse, expected: dict | None
    ):
        assert self.stream._extract_usage(chunk) == expected


class TestPayloadHelpers:
    """`_extract_text_payload` and `_extract_thought_payload` are the
    cross-provider routers used by `OpenAIChatStream` and
    `_OpenAIBase.complete_chat`. They have to handle three wire shapes:
    OpenAI/Grok plain string, Mistral structured chunk arrays, and the
    Grok/DeepSeek `reasoning_content` sibling field.
    """

    @pytest.mark.parametrize(
        "payload, expected",
        [
            pytest.param({"content": "hi"}, "hi", id="plain-string"),
            pytest.param({"content": ""}, None, id="empty-string"),
            pytest.param({"content": None}, None, id="null"),
            pytest.param({}, None, id="missing"),
            pytest.param(
                {
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"type": "text", "text": "world"},
                    ]
                },
                "Hello world",
                id="mistral-text-chunks",
            ),
            pytest.param(
                {
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [{"type": "text", "text": "ignore me"}],
                        },
                        {"type": "text", "text": "the answer"},
                    ]
                },
                "the answer",
                id="mistral-mixed-skips-thinking",
            ),
            pytest.param(
                {
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [{"type": "text", "text": "only thinking"}],
                        }
                    ]
                },
                None,
                id="mistral-thinking-only",
            ),
        ],
    )
    def test_extract_text_payload(self, payload: dict, expected: str | None):
        assert _extract_text_payload(payload) == expected

    @pytest.mark.parametrize(
        "payload, expected",
        [
            pytest.param(
                {"reasoning_content": "thinking..."}, "thinking...", id="grok"
            ),
            pytest.param({"reasoning_content": ""}, None, id="grok-empty"),
            pytest.param({"content": "just text"}, None, id="plain-no-thought"),
            pytest.param(
                {
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [
                                {"type": "text", "text": "first "},
                                {"type": "text", "text": "second"},
                            ],
                        },
                        {"type": "text", "text": "answer"},
                    ]
                },
                "first second",
                id="mistral-thinking",
            ),
            pytest.param(
                {
                    "content": [
                        {"type": "text", "text": "no thinking here"},
                    ]
                },
                None,
                id="mistral-text-only",
            ),
            pytest.param({}, None, id="missing"),
        ],
    )
    def test_extract_thought_payload(self, payload: dict, expected: str | None):
        assert _extract_thought_payload(payload) == expected


class TestStreamForwardsThoughts:
    """`OpenAIChatStream.__aiter__` must forward reasoning chunks to
    `client.on_thought` for both Grok-style `reasoning_content` and
    Mistral-style `ThinkChunk` payloads, while keeping the regular text
    stream free of any thought content. When `on_thought` is unset,
    reasoning chunks are silently dropped — never raised, never yielded.
    """

    @pytest.mark.parametrize(
        "chunks, expected_text, expected_thoughts, with_callback",
        [
            pytest.param(
                [
                    {"choices": [{"delta": {"reasoning_content": "Let me think... "}}]},
                    {"choices": [{"delta": {"reasoning_content": "ok done."}}]},
                    {"choices": [{"delta": {"content": "The answer is 42"}}]},
                ],
                "The answer is 42",
                ["Let me think... ", "ok done."],
                True,
                id="grok_reasoning_content",
            ),
            pytest.param(
                [{"choices": [{"delta": {"content": "The answer is 42"}}]}],
                "The answer is 42",
                [],
                True,
                id="reasoning_not_exposed",
            ),
            pytest.param(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": [
                                        {
                                            "type": "thinking",
                                            "thinking": [
                                                {
                                                    "type": "text",
                                                    "text": "Reasoning step 1",
                                                }
                                            ],
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": [
                                        {"type": "text", "text": "Final answer."},
                                    ]
                                }
                            }
                        ]
                    },
                ],
                "Final answer.",
                ["Reasoning step 1"],
                True,
                id="mistral_thinking_chunks",
            ),
            pytest.param(
                [
                    {
                        "choices": [
                            {"delta": {"reasoning_content": "internal scratchpad"}}
                        ]
                    },
                    {"choices": [{"delta": {"content": "answer"}}]},
                ],
                "answer",
                [],
                False,  # on_thought=None
                id="no_callback_drops_silently",
            ),
        ],
    )
    async def test_forwards_or_drops_thoughts(
        self,
        chunks: list[dict],
        expected_text: str,
        expected_thoughts: list[str],
        with_callback: bool,
    ):
        async def fake_stream(_body):
            for c in chunks:
                yield c

        thoughts: list[str] = []
        client = MagicMock(spec=OpenAIClient)
        client.model = "test-model"
        client.temperature = 0.2
        client.provider = "openai"
        client.on_thought = thoughts.append if with_callback else None
        client.stream = fake_stream
        stream = OpenAIChatStream(client, [])
        produced = [t async for t in stream]
        assert "".join(produced) == expected_text
        assert thoughts == expected_thoughts


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(
            {
                "content": [
                    {
                        "type": "thinking",
                        "thinking": [{"type": "text", "text": "Thinking"}],
                    },
                    {"type": "text", "text": "56"},
                ]
            },
            id="mistral",
        ),
        pytest.param({"content": "56", "reasoning_content": "Thinking"}, id="grok"),
    ],
)
async def test_complete_chat_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch, message: dict
):
    thoughts: list[str] = []
    client = OpenAIClient(api_key="k", on_thought=thoughts.append)
    complete = AsyncMock(
        return_value=(
            {
                "choices": [
                    {
                        "message": message,
                        "finish_reason": "stop",
                    }
                ],
            },
            {"input": 1, "output": 2, "total": 3},
        )
    )
    monkeypatch.setattr(client, "complete", complete)

    response, _ = await client.complete_chat(
        [{"role": "user", "content": "What is 7 * 8?"}],
        extra_params={"reasoning_effort": "high"},
    )

    assert complete.call_args.args[0]["reasoning_effort"] == "high"
    assert response["content"] == "56"
    assert thoughts == ["Thinking"]


class TestOpenAIStream:
    """Tests for _OpenAIBase.stream() SSE parsing."""

    @pytest.fixture
    def client(self):
        c = MagicMock(spec=OpenAIClient)
        c.provider = "openai"
        c.model = "gpt-4o"
        c.base_url = "https://api.openai.com/v1/"
        c._sse_url = OpenAIClient._sse_url.__get__(c)
        c._iter_sse = OpenAIClient._iter_sse.__get__(c)
        c.session = AsyncMock()
        return c

    @pytest.mark.asyncio
    async def test_yields_parsed_chunks(self, client, make_sse_event, make_sse_resp):
        chunk1 = {"choices": [{"delta": {"content": "Hello"}}]}
        chunk2 = {"choices": [{"delta": {"content": " world"}}]}
        events = [
            make_sse_event(json.dumps(chunk1)),
            make_sse_event(json.dumps(chunk2)),
        ]
        client.session.post.return_value = make_sse_resp(events)
        chunks = [
            c
            async for c in OpenAIClient.stream(
                client, {"model": "gpt-4o", "messages": []}
            )
        ]
        assert chunks == [chunk1, chunk2]

    @pytest.mark.asyncio
    async def test_done_sentinel_stops(self, client, make_sse_event, make_sse_resp):
        chunk = {"choices": [{"delta": {"content": "hi"}}]}
        done_ev = MagicMock()
        done_ev.data = "[DONE]"
        done_ev.json = MagicMock(side_effect=ValueError)
        events = [
            make_sse_event(json.dumps(chunk)),
            done_ev,
            make_sse_event(json.dumps({"should": "not appear"})),
        ]
        client.session.post.return_value = make_sse_resp(events)
        chunks = [
            c
            async for c in OpenAIClient.stream(
                client, {"model": "gpt-4o", "messages": []}
            )
        ]
        assert chunks == [chunk]

    @pytest.mark.asyncio
    async def test_cleanup_failure_does_not_corrupt_stream(
        self, client, make_sse_event, make_sse_resp
    ):
        """Aborting an already-finished HTTP/2 stream raises KeyError inside
        urllib3-future; the consumer must still get a clean stream end and the
        connection must still be torn down and released."""
        chunk = {"choices": [{"delta": {"content": "hi"}}]}
        done_ev = MagicMock()
        done_ev.data = "[DONE]"
        done_ev.json = MagicMock(side_effect=ValueError)
        resp = make_sse_resp([make_sse_event(json.dumps(chunk)), done_ev])

        async def _h2_abort_quirk():
            raise KeyError(1)

        resp.extension.close = _h2_abort_quirk
        client.session.post.return_value = resp
        chunks = [
            c
            async for c in OpenAIClient.stream(
                client, {"model": "gpt-4o", "messages": []}
            )
        ]
        assert chunks == [chunk]
        resp.raw.close.assert_awaited_once()
        resp.raw.release_conn.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_keepalive_frames(self, client, make_sse_event, make_sse_resp):
        chunk = {"choices": [{"delta": {"content": "ok"}}]}
        events = [
            make_sse_event(""),
            make_sse_event(json.dumps(chunk)),
            make_sse_event(""),
        ]
        client.session.post.return_value = make_sse_resp(events)
        chunks = [
            c
            async for c in OpenAIClient.stream(
                client, {"model": "gpt-4o", "messages": []}
            )
        ]
        assert chunks == [chunk]

    @pytest.mark.asyncio
    async def test_no_extension_raises(self, client):
        resp = AsyncMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock(return_value=resp)
        resp.extension = None
        client.session.post.return_value = resp
        with pytest.raises(LLMError, match="SSE extension"):
            _ = [
                c
                async for c in OpenAIClient.stream(
                    client, {"model": "gpt-4o", "messages": []}
                )
            ]

    @pytest.mark.asyncio
    async def test_malformed_json_raises(self, client, make_sse_resp):
        ev = MagicMock()
        ev.data = "{bad json"
        ev.json = MagicMock(side_effect=ValueError("parse error"))
        client.session.post.return_value = make_sse_resp([ev])
        with pytest.raises(LLMError, match="Stream parse error"):
            _ = [
                c
                async for c in OpenAIClient.stream(
                    client, {"model": "gpt-4o", "messages": []}
                )
            ]


class TestOpenAIStreamRaw:
    """Tests for _OpenAIBase.stream_raw(): the provider's SSE bytes, relayed unparsed."""

    SSE = (
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
        b"data: [DONE]\n\n",
    )

    @pytest.fixture
    def client(self):
        c = MagicMock(spec=OpenAIClient)
        c.provider = "openai"
        c.session = AsyncMock()
        return c

    @staticmethod
    def _resp(status: int = 200, chunks: tuple[bytes, ...] = (), body=None):
        resp = AsyncMock()
        resp.status_code = status
        resp.headers = {}
        if status >= 400:
            resp.raise_for_status = MagicMock(side_effect=HTTPError(response=resp))
        else:
            resp.raise_for_status = MagicMock(return_value=resp)

        async def _chunks():
            for chunk in chunks:
                yield chunk

        resp.iter_content = AsyncMock(return_value=_chunks())
        resp.json = AsyncMock(return_value=body)
        return resp

    @pytest.mark.asyncio
    async def test_relays_bytes_unchanged_then_closes(self, client):
        resp = self._resp(chunks=self.SSE)
        client.session.post.return_value = resp
        headers = {"x-litellm-end-user-id": "42"}
        relayed = [
            c
            async for c in OpenAIClient.stream_raw(
                client, {"model": "gpt-4o", "messages": []}, headers=headers
            )
        ]
        assert relayed == list(self.SSE)
        kwargs = client.session.post.call_args.kwargs
        assert kwargs["json"] == {"model": "gpt-4o", "messages": [], "stream": True}
        assert kwargs["headers"] == headers
        assert kwargs["stream"] is True
        # A positive chunk size buffers the whole reply before the first yield.
        resp.iter_content.assert_awaited_once_with(-1)
        resp.close.assert_awaited_once()

    @pytest.mark.parametrize(
        "close_error",
        [
            pytest.param(None, id="clean-close"),
            pytest.param(KeyError(1), id="http2-abort-raises"),
        ],
    )
    async def test_closes_when_the_caller_stops_reading(self, client, close_error):
        resp = self._resp(chunks=self.SSE)
        resp.close.side_effect = close_error
        client.session.post.return_value = resp
        async with aclosing(
            OpenAIClient.stream_raw(client, {"model": "gpt-4o", "messages": []})
        ) as stream:
            async for _ in stream:
                break
        resp.close.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status, body, error",
        [
            pytest.param(429, {"error": "slow down"}, TooManyRequestsError, id="429"),
            pytest.param(402, {"error": "no credit"}, QuotaExceededError, id="402"),
            pytest.param(500, {"error": {"message": "boom"}}, LLMError, id="500"),
        ],
    )
    async def test_failure_raises_before_the_first_byte(
        self, client, status, body, error
    ):
        resp = self._resp(status=status, chunks=self.SSE, body=body)
        client.session.post.return_value = resp
        relayed: list[bytes] = []
        with pytest.raises(error):
            async for chunk in OpenAIClient.stream_raw(
                client, {"model": "gpt-4o", "messages": []}
            ):
                relayed.append(chunk)
        assert relayed == []
        resp.iter_content.assert_not_called()
        resp.close.assert_awaited_once()


_COMPLETION = {
    "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
    "usage": {"total_tokens": 3, "prompt_tokens": 2, "completion_tokens": 1},
}


async def _call_complete(client: OpenAIClient, headers):
    await client.complete({"model": "gpt-4o", "messages": []}, headers=headers)


async def _call_complete_chat(client: OpenAIClient, headers):
    await client.complete_chat([{"role": "user", "content": "hi"}], headers=headers)


async def _call_stream(client: OpenAIClient, headers):
    _ = [
        c
        async for c in client.stream(
            {"model": "gpt-4o", "messages": []}, headers=headers
        )
    ]


async def _call_stream_chat(client: OpenAIClient, headers):
    _ = [
        c
        async for c in client.stream_chat(
            [{"role": "user", "content": "hi"}], headers=headers
        )
    ]


@pytest.mark.parametrize(
    "call, streams",
    [
        pytest.param(_call_complete, False, id="complete"),
        pytest.param(_call_complete_chat, False, id="complete_chat"),
        pytest.param(_call_stream, True, id="stream"),
        pytest.param(_call_stream_chat, True, id="stream_chat"),
    ],
)
async def test_sends_per_call_headers(call, streams, make_resp, make_sse_resp):
    client = OpenAIClient(model="gpt-4o", api_key="k")
    client._session = AsyncMock()
    client._session.post.return_value = (
        make_sse_resp([]) if streams else make_resp(200, _COMPLETION)
    )
    await call(client, MappingProxyType({"x-litellm-end-user-id": "42"}))
    assert client._session.post.call_args.kwargs["headers"] == {
        "x-litellm-end-user-id": "42"
    }


class TestOpenAITranscribe:
    """Tests for OpenAIClient.transcribe()."""

    @pytest.fixture
    def client(self, make_resp):
        c = MagicMock(spec=OpenAIClient)
        c.provider = "openai"
        c.session = AsyncMock()
        c.session.post.return_value = make_resp(200, {"text": "deux palettes"})
        return c

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kwargs, expected_file, expected_data",
        [
            pytest.param(
                {"filename": "note.mp3"},
                ("note.mp3", b"RIFF", "audio/mpeg"),
                [("model", "whisper-1")],
                id="bytes-mime-from-name",
            ),
            pytest.param(
                {
                    "filename": "blob",
                    "content_type": "audio/ogg",
                    "model": "whisper-large-v3",
                    "language": "fr",
                    "prompt": "Soren, Loxam",
                    "temperature": 0.0,
                },
                ("blob", b"RIFF", "audio/ogg"),
                [
                    ("model", "whisper-large-v3"),
                    ("language", "fr"),
                    ("prompt", "Soren, Loxam"),
                    ("temperature", "0.0"),
                ],
                id="every-field",
            ),
        ],
    )
    async def test_posts_multipart(self, client, kwargs, expected_file, expected_data):
        headers = {"x-litellm-end-user-id": "42"}
        result = await OpenAIClient.transcribe(
            client, b"RIFF", headers=headers, **kwargs
        )
        assert result == {"text": "deux palettes"}
        args, call = client.session.post.call_args
        assert args == ("/audio/transcriptions",)
        assert call["files"] == [("file", expected_file)]
        assert call["data"] == expected_data
        assert call["headers"] == headers

    @pytest.mark.asyncio
    async def test_reads_a_path(self, client, tmp_path):
        path = tmp_path / "memo.mp3"
        path.write_bytes(b"ID3")
        await OpenAIClient.transcribe(client, path)
        assert client.session.post.call_args.kwargs["files"] == [
            ("file", ("memo.mp3", b"ID3", "audio/mpeg"))
        ]

    @pytest.mark.asyncio
    async def test_raw_bytes_need_a_filename(self, client):
        with pytest.raises(LLMError, match="filename"):
            await OpenAIClient.transcribe(client, b"RIFF")
        client.session.post.assert_not_called()


class TestBatchResultFromLine:
    @pytest.mark.parametrize(
        "data, expected",
        [
            pytest.param(
                {
                    "custom_id": "req-1",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [{"message": {"content": "Hello!"}, "index": 0}],
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 5,
                                "total_tokens": 15,
                            },
                        },
                    },
                },
                BatchResult(
                    custom_id="req-1",
                    content="Hello!",
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                ),
                id="full-response",
            ),
            pytest.param(
                {
                    "custom_id": "req-2",
                    "response": {
                        "status_code": 200,
                        "body": {"choices": [], "usage": {}},
                    },
                },
                BatchResult(custom_id="req-2", content=""),
                id="empty-choices",
            ),
            pytest.param(
                {
                    "custom_id": "req-3",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [{"message": {"content": None}, "index": 0}],
                            "usage": {"prompt_tokens": 5, "total_tokens": 5},
                        },
                    },
                },
                BatchResult(
                    custom_id="req-3",
                    content="",
                    input_tokens=5,
                    total_tokens=5,
                ),
                id="null-content",
            ),
            pytest.param(
                {"custom_id": "req-4", "error": {"code": "server_error"}},
                BatchResult(custom_id="req-4", content=""),
                id="error-response",
            ),
        ],
    )
    def test_from_line(self, data: dict, expected: BatchResult):
        assert BatchResult.from_line(data) == expected


class TestBatchJobProperties:
    @pytest.mark.parametrize(
        "status, expected",
        [
            pytest.param("completed", True, id="completed"),
            pytest.param("failed", True, id="failed"),
            pytest.param("expired", True, id="expired"),
            pytest.param("cancelled", True, id="cancelled"),
            pytest.param("in_progress", False, id="in_progress"),
            pytest.param("validating", False, id="validating"),
            pytest.param("finalizing", False, id="finalizing"),
            pytest.param("cancelling", False, id="cancelling"),
        ],
    )
    def test_is_terminal(self, status: str, expected: bool):
        job = BatchJob(
            id="batch_123",
            status=status,  # pyright: ignore[reportArgumentType]
            endpoint="/v1/chat/completions",
            input_file_id="file-abc",
        )
        assert job.is_terminal is expected

    @pytest.mark.parametrize(
        "status, expected",
        [
            pytest.param("completed", True, id="completed"),
            pytest.param("failed", False, id="failed"),
            pytest.param("in_progress", False, id="in_progress"),
        ],
    )
    def test_succeeded(self, status: str, expected: bool):
        job = BatchJob(
            id="batch_123",
            status=status,  # pyright: ignore[reportArgumentType]
            endpoint="/v1/chat/completions",
            input_file_id="file-abc",
        )
        assert job.succeeded is expected
