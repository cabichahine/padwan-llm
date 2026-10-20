from contextlib import nullcontext

import pytest

from padwan_ai import OpenAIClient
from padwan_ai.openai import BatchRequest

from .conftest import AUDIO_FIXTURE, skip_no_openai

pytestmark = [pytest.mark.e2e, skip_no_openai]


async def test_batch_lifecycle() -> None:
    async with OpenAIClient() as client:
        req = BatchRequest(
            body={
                "messages": [{"role": "user", "content": "Say hello"}],
                "model": "gpt-4o-mini",
            },
            custom_id="e2e-openai-1",
        )
        job = await client.create_batch([req], model="gpt-4o-mini")
        assert job.id
        assert not job.is_terminal

        fetched = await client.get_batch(job.id)
        assert fetched.id == job.id

        jobs, _ = await client.list_batches(limit=5)
        assert any(j.id == job.id for j in jobs)

        cancelled = await client.cancel_batch(job.id)
        assert cancelled.id == job.id


async def test_embeddings() -> None:
    async with OpenAIClient(model="text-embedding-3-small") as client:
        resp = await client.fetch_embeddings(["Hello", "World"], dimensions=256)
        assert len(resp["data"]) == 2
        assert len(resp["data"][0]["embedding"]) == 256
        assert resp["usage"]["total_tokens"] > 0


@pytest.mark.skipif(not AUDIO_FIXTURE.exists(), reason="audio fixture not found")
@pytest.mark.parametrize(
    "as_bytes, filename, ctx",
    [
        pytest.param(False, None, nullcontext(), id="path"),
        pytest.param(True, "clip.wav", nullcontext(), id="bytes"),
        pytest.param(
            True,
            None,
            pytest.raises(ValueError, match="filename"),
            id="bytes-without-filename",
        ),
    ],
)
async def test_transcribe(as_bytes, filename, ctx) -> None:
    file = AUDIO_FIXTURE.read_bytes() if as_bytes else AUDIO_FIXTURE
    async with OpenAIClient() as client:
        with ctx:
            resp = await client.transcribe(file=file, filename=filename)
            assert "pallet" in resp["text"].lower()
