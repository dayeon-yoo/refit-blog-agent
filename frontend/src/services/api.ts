import type { BlogIdea, IdeaCandidate, IdeaExpansionResult, WorkflowResult } from '../types/models';

const baseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

export type GenerationMode = 'mock' | 'live' | 'unknown';

async function post<T>(path: string, payload: unknown, onMode?: (mode: GenerationMode) => void): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${baseUrl}${path}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
  } catch {
    throw new Error('서버에 연결할 수 없습니다. 백엔드 실행 상태와 연결 주소를 확인한 뒤 다시 시도해주세요.');
  }
  const mode = response.headers.get('X-RIFIT-LLM-Mode');
  if (!response.ok) {
    if (response.status === 422) throw new Error('입력 또는 생성 결과를 확인하는 과정에서 멈췄습니다. 내용을 검토하거나 같은 단계에서 다시 시도해주세요.');
    throw new Error('요청을 완료하지 못했습니다. 입력은 유지되어 있으니 잠시 후 다시 시도해주세요.');
  }
  try {
    const data = await response.json() as T;
    onMode?.(mode === 'mock' || mode === 'live' ? mode : 'unknown');
    return data;
  }
  catch { throw new Error('서버 응답을 읽지 못했습니다. 잠시 후 다시 시도해주세요.'); }
}

export const expandIdea = (source: string, onMode?: (mode: GenerationMode) => void, useLive = false) =>
  post<IdeaExpansionResult>('/api/ideas/expand', useLive ? { source, use_live: true } : { source }, onMode);
export const refineIdea = (source: string, candidate: IdeaCandidate, revision_request: string) =>
  post<BlogIdea>('/api/ideas/refine', { source, candidate, revision_request });
export const generateWorkflow = (source: string, blog_idea: BlogIdea) =>
  post<WorkflowResult>('/api/workflow/generate', { source, blog_idea });
