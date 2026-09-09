import { afterEach, describe, expect, it, vi } from 'vitest';
import { expandIdea, refineIdea } from './api';

afterEach(() => vi.unstubAllGlobals());
describe('API boundary', () => {
  it('requests live generation only with explicit opt-in', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ candidates: [] })));
    vi.stubGlobal('fetch', fetch);
    await expandIdea('주제', undefined, true);
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ source: '주제', use_live: true });
    expect(fetch).toHaveBeenCalledTimes(1);
  });
  it.each(['mock', 'live', 'unknown'] as const)('reports %s mode without an extra request', async mode => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ source_input: '원문', candidates: [] }), {
      headers: mode === 'unknown' ? {} : { 'X-RIFIT-LLM-Mode': mode },
    }));
    vi.stubGlobal('fetch', fetch);
    const onMode = vi.fn();
    await expandIdea('원문', onMode);
    expect(onMode).toHaveBeenCalledWith(mode);
    expect(fetch).toHaveBeenCalledTimes(1);
  });
  it('sends the exact selected candidate and revision', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ title: '결과' })));
    vi.stubGlobal('fetch', fetch);
    const candidate = { candidate_id: '2', title: '제목', perspective: '관점', content_format: '가이드', key_question: '질문', brief_description: '설명' };
    await refineIdea('원문', candidate, '요청');
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0][0]).toBe('http://localhost:8000/api/ideas/refine');
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ source: '원문', candidate, revision_request: '요청' });
  });
  it('never exposes validation internals', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: 'SourceIntent traceback private' }), { status: 422 })));
    await expect(expandIdea('메모')).rejects.toThrow('입력 또는 생성 결과');
  });
  it('does not relabel retained mock candidates as live after a failed paid request', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 500, headers: { 'X-RIFIT-LLM-Mode': 'live' } })));
    const onMode = vi.fn();
    await expect(expandIdea('원문', onMode, true)).rejects.toThrow();
    expect(onMode).not.toHaveBeenCalled();
  });
  it('handles network failures without discarding the request', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    await expect(expandIdea('메모')).rejects.toThrow('서버에 연결할 수 없습니다');
  });
});
