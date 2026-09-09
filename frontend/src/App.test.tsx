import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import App from './App';
import * as api from './services/api';
import type { BlogIdea, WorkflowResult } from './types/models';

vi.mock('./services/api');
let host: HTMLDivElement;
let root: Root;
const candidate = { candidate_id: '2', title: '메모에서 나온 제목', perspective: '관점', content_format: '가이드', key_question: '무엇을 소개할까요?', brief_description: '후보 설명' };
const idea: BlogIdea = { title: '확정 제목', keyword: '키워드', summary: '요약', search_intent: 'INTERNAL_SEARCH', angle: '내부 angle', rifit_connection: '', seasonality: 0, outline: ['전개 하나'] };
const result: WorkflowResult = {
  blog_idea: idea, content_plan: { writing_script: 'INTERNAL_SCRIPT' },
  post: { title: '최종 제목', content: '## 본문 소제목\n\n읽을 수 있는 본문', summary: '요약', keyword: '키워드', cta: '마무리' },
  tags: ['태그하나', '태그둘'], image_plan: { images: [{ placement: '도입', purpose: '주제 소개', prompt: '이미지 구상', alt_text: '이미지 설명', image_type: '사진' }] },
  qc: { status: 'BLOCK', summary: '표현을 확인해주세요', issues: [{ category: 'grounding', severity: 'error', message: '근거를 확인해주세요', evidence: '문제 문장', suggestion: '표현 수정' }] },
};

beforeEach(async () => {
  vi.resetAllMocks();
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  await act(async () => root.render(<App />));
});
afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.unstubAllGlobals(); });

function button(text: string): HTMLButtonElement {
  const found = Array.from(host.querySelectorAll('button')).find(item => item.textContent?.includes(text));
  if (!found) throw new Error(`Missing button: ${text}`);
  return found;
}
async function click(text: string) { await act(async () => button(text).click()); }
async function input(id: string, value: string) {
  await act(async () => {
    const field = host.querySelector<HTMLTextAreaElement>(`#${id}`)!;
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')!.set!.call(field, value);
    field.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

async function reachResult() {
  vi.mocked(api.expandIdea).mockResolvedValue({ source_input: '원문', candidates: [candidate] });
  vi.mocked(api.refineIdea).mockResolvedValue(idea);
  vi.mocked(api.generateWorkflow).mockResolvedValue(result);
  await input('source', '원문');
  await click('아이디어 확장하기');
  await click('이 아이디어 선택');
  await input('revision', '보존할 요청');
  await click('아이디어 확정하기');
  await click('이 내용으로 글 생성');
}

function stepButtons() { return Array.from(host.querySelectorAll<HTMLButtonElement>('.steps button')); }

it('labels mock candidates rather than presenting them as real AI proposals', async () => {
  vi.mocked(api.expandIdea).mockImplementation(async (source, onMode) => {
    onMode?.('mock');
    return { source_input: source, candidates: [candidate] };
  });
  await input('source', '입력한 주제');
  await click('아이디어 확장하기');
  expect(host.textContent).toContain('현재는 화면 확인용 샘플입니다');
  expect(host.textContent).toContain(candidate.title);
  expect(api.expandIdea).toHaveBeenCalledTimes(1);
  expect(api.refineIdea).not.toHaveBeenCalled();
  expect(api.generateWorkflow).not.toHaveBeenCalled();
});

it('explicitly generates live candidates once and shows subtitle and direction instead of an internal question', async () => {
  vi.mocked(api.expandIdea).mockImplementationOnce(async (source, onMode) => {
    onMode?.('mock');
    return { source_input: source, candidates: [candidate] };
  });
  await input('source', '출근 준비 아이디어');
  await click('아이디어 확장하기');
  const live = { ...candidate, title: '출근 준비를 가볍게 만드는 네 가지 방법', perspective: '옷 선택부터 정리하는 아침 준비', brief_description: '출근 전 옷 선택 과정을 중심으로 네 가지 방법을 구성한다.' };
  vi.mocked(api.expandIdea).mockImplementationOnce(async (source, onMode) => {
    onMode?.('live');
    return { source_input: source, candidates: [live] };
  });
  await click('실제 AI로 후보 3개 생성');
  expect(api.expandIdea).toHaveBeenLastCalledWith('출근 준비 아이디어', expect.any(Function), true);
  expect(api.expandIdea).toHaveBeenCalledTimes(2);
  expect(host.textContent).toContain(live.title);
  expect(host.textContent).toContain(live.perspective);
  expect(host.textContent).toContain(live.brief_description);
  expect(host.textContent).not.toContain(live.key_question);
  expect(api.refineIdea).not.toHaveBeenCalled();
  expect(api.generateWorkflow).not.toHaveBeenCalled();
});

it('shows a text brand if the optional logo image is unavailable', async () => {
  const logo = host.querySelector<HTMLImageElement>('.brand-logo')!;
  expect(logo.getAttribute('src')).toBe('/refit-logo.png');
  await act(async () => logo.dispatchEvent(new Event('error')));
  expect(host.querySelector('.brand-fallback')?.textContent).toBe('RIFIT');
  expect(host.querySelector('.brand-logo')).toBeNull();
});

it('moves back exactly one step while keeping source, selection and revision without API calls', async () => {
  expect(host.textContent).not.toContain('이전 단계');
  expect(stepButtons().every(item => item.disabled)).toBe(true);
  await reachResult();
  expect(stepButtons()[4].getAttribute('aria-current')).toBe('step');
  expect(stepButtons().slice(0, 4).every(item => !item.disabled)).toBe(true);
  await click('이전 단계');
  expect(host.querySelector('.page-heading h2')?.textContent).toBe('글 생성');
  expect(host.textContent).toContain(idea.title);
  expect(stepButtons()[4].disabled).toBe(true);
  await act(async () => stepButtons()[4].click());
  expect(host.querySelector('.page-heading h2')?.textContent).toBe('글 생성');
  await click('이전 단계');
  expect(host.querySelector<HTMLTextAreaElement>('#revision')?.value).toBe('보존할 요청');
  await click('이전 단계');
  expect(host.querySelector('[aria-pressed=true]')).not.toBeNull();
  await click('이전 단계');
  expect(host.querySelector<HTMLTextAreaElement>('#source')?.value).toBe('원문');
  expect(host.textContent).not.toContain('이전 단계');
  expect(api.expandIdea).toHaveBeenCalledTimes(1);
  expect(api.refineIdea).toHaveBeenCalledTimes(1);
  expect(api.generateWorkflow).toHaveBeenCalledTimes(1);
});

it('allows backward stepper navigation after an error and clears only the transient error', async () => {
  await reachResult();
  await act(async () => stepButtons()[3].click());
  vi.mocked(api.generateWorkflow).mockRejectedValueOnce(new Error('재생성 실패'));
  await click('이 내용으로 글 생성');
  expect(host.querySelector('[role=alert]')?.textContent).toContain('재생성 실패');
  await act(async () => stepButtons()[2].click());
  expect(host.querySelector('[role=alert]')).toBeNull();
  expect(host.querySelector<HTMLTextAreaElement>('#revision')?.value).toBe('보존할 요청');
  expect(stepButtons()[3].disabled).toBe(true);
  await act(async () => stepButtons()[1].click());
  expect(host.querySelector('[aria-pressed=true]')).not.toBeNull();
  expect(api.expandIdea).toHaveBeenCalledTimes(1);
  expect(api.refineIdea).toHaveBeenCalledTimes(1);
  expect(api.generateWorkflow).toHaveBeenCalledTimes(2);
});

it('uses the new candidate and revision on explicit reruns without displaying stale downstream results', async () => {
  const other = { ...candidate, candidate_id: '3', title: '다른 후보' };
  vi.mocked(api.expandIdea).mockResolvedValue({ source_input: '원문', candidates: [candidate, other] });
  vi.mocked(api.refineIdea).mockResolvedValue(idea);
  vi.mocked(api.generateWorkflow).mockResolvedValue(result);
  await input('source', '원문'); await click('아이디어 확장하기'); await click('이 아이디어 선택');
  await click('아이디어 확정하기'); await click('이 내용으로 글 생성');
  await act(async () => stepButtons()[1].click());
  await act(async () => host.querySelectorAll<HTMLButtonElement>('.select-button')[1].click());
  expect(host.textContent).not.toContain('최종 제목');
  expect(stepButtons()[3].disabled).toBe(true);
  await input('revision', '새 요청');
  vi.mocked(api.refineIdea).mockResolvedValueOnce({ ...idea, title: '새 확정 제목' });
  await click('아이디어 확정하기');
  expect(api.refineIdea).toHaveBeenLastCalledWith('원문', other, '새 요청');
  expect(host.textContent).toContain('새 확정 제목');
  expect(host.textContent).not.toContain('최종 제목');
  await act(async () => stepButtons()[0].click());
  await input('source', '변경된 원문');
  vi.mocked(api.expandIdea).mockResolvedValueOnce({ source_input: '변경된 원문', candidates: [other] });
  await click('아이디어 확장하기');
  expect(api.expandIdea).toHaveBeenLastCalledWith('변경된 원문', expect.any(Function));
  expect(host.querySelector('[aria-pressed=true]')).toBeNull();
  expect(host.textContent).not.toContain(candidate.title);
});
it('preserves input on errors, blocks double requests, and retries the same step', async () => {
  let reject!: (reason: Error) => void;
  vi.mocked(api.expandIdea).mockImplementationOnce(() => new Promise((_, fail) => { reject = fail; }));
  await input('source', '계속 보존할 원본 메모');
  await click('아이디어 확장하기');
  expect(button('아이디어 확장하기').disabled).toBe(true);
  await click('아이디어 확장하기');
  expect(api.expandIdea).toHaveBeenCalledTimes(1);
  await act(async () => reject(new Error('다시 시도해주세요')));
  expect(host.querySelector('[role=alert]')?.textContent).toContain('다시 시도');
  expect(host.querySelector<HTMLTextAreaElement>('#source')?.value).toBe('계속 보존할 원본 메모');
  expect(button('아이디어 확장하기').disabled).toBe(false);
});
it('completes selected-candidate flow, keeps BLOCK content visible, copies tags and returns to selection', async () => {
  vi.mocked(api.expandIdea).mockResolvedValue({ source_input: '원문', candidates: [candidate] });
  vi.mocked(api.refineIdea).mockResolvedValue(idea);
  vi.mocked(api.generateWorkflow).mockRejectedValueOnce(new Error('생성 실패')).mockResolvedValueOnce(result);
  const clipboard = vi.fn().mockResolvedValue(undefined);
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: clipboard } });
  await input('source', '원문'); await click('아이디어 확장하기'); await click('이 아이디어 선택');
  await input('revision', '집중할 방향'); await click('아이디어 확정하기');
  expect(api.refineIdea).toHaveBeenCalledWith('원문', candidate, '집중할 방향');
  expect(host.textContent).not.toContain('INTERNAL_SEARCH');
  await click('이 내용으로 글 생성');
  expect(host.textContent).toContain('생성 실패');
  expect(host.textContent).toContain('확정 제목');
  await click('이 내용으로 글 생성');
  expect(api.generateWorkflow).toHaveBeenLastCalledWith('원문', idea);
  expect(host.textContent).toContain('수정 필요'); expect(host.textContent).toContain('읽을 수 있는 본문');
  expect(host.textContent).not.toContain('INTERNAL_SCRIPT');
  await click('태그 복사'); expect(clipboard).toHaveBeenCalledWith('#태그하나 #태그둘');
  await click('본문 복사'); expect(clipboard).toHaveBeenLastCalledWith('최종 제목\n\n## 본문 소제목\n\n읽을 수 있는 본문\n\n마무리');
  await click('후보 선택으로'); expect(host.querySelector('[aria-pressed=true]')).not.toBeNull();
  await click('이 아이디어 선택'); expect(host.querySelector<HTMLTextAreaElement>('#revision')?.value).toBe('집중할 방향');
  await click('원본 메모 보기'); expect(host.querySelector<HTMLTextAreaElement>('#source')?.value).toBe('원문');
});
