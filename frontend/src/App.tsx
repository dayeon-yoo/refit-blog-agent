import { useRef, useState } from 'react';
import { expandIdea as requestExpansion, refineIdea, generateWorkflow } from './services/api';
import type { GenerationMode } from './services/api';
import type { BlogIdea, IdeaCandidate, WorkflowResult } from './types/models';
import ResultView from './components/ResultView';
import Header from './components/Header';
import CandidateSelection from './components/CandidateSelection';
import './workflow-navigation.css';

type WorkflowStep = 1 | 2 | 3 | 4 | 5;
const steps: { id: WorkflowStep; label: string }[] = [
  { id: 1, label: '아이디어 입력' },
  { id: 2, label: '후보 선택' },
  { id: 3, label: '방향 다듬기' },
  { id: 4, label: '글 생성' },
  { id: 5, label: '결과 확인' },
];
const previousStep: Record<WorkflowStep, WorkflowStep> = { 1: 1, 2: 1, 3: 2, 4: 3, 5: 4 };

export default function App() {
  const [step, setStep] = useState<WorkflowStep>(1);
  const [source, setSource] = useState('');
  const [candidates, setCandidates] = useState<IdeaCandidate[]>([]);
  const [selected, setSelected] = useState<IdeaCandidate | null>(null);
  const [revision, setRevision] = useState('');
  const [idea, setIdea] = useState<BlogIdea | null>(null);
  const [result, setResult] = useState<WorkflowResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [generationMode, setGenerationMode] = useState<GenerationMode>('unknown');
  const expandIdea = (input: string) => requestExpansion(input, setGenerationMode);
  const lock = useRef(false);
  const heading = useRef<HTMLHeadingElement>(null);
  function go(next: WorkflowStep) { setError(''); setStep(next); requestAnimationFrame(() => heading.current?.focus()); }
  async function run(action: () => Promise<void>) {
    if (lock.current) return;
    lock.current = true; setBusy(true); setError('');
    try { await action(); }
    catch (e) { setError(e instanceof Error ? e.message : '요청을 완료하지 못했습니다. 다시 시도해주세요.'); }
    finally { lock.current = false; setBusy(false); }
  }
  function reset() {
    setSource(''); setCandidates([]); setSelected(null); setRevision(''); setIdea(null); setResult(null); go(1);
  }
  return <div className="workspace">
    <Header />
    <nav aria-label="작성 단계"><ol className="steps">{steps.map(({ id, label }) =>
      <li key={id} className={step === id ? 'active' : step > id ? 'complete' : ''}>
        <button type="button" disabled={busy || id >= step} aria-current={step === id ? 'step' : undefined} onClick={() => go(id)}>
          <span>{String(id).padStart(2, '0')}</span>{label}
        </button>
      </li>
    )}</ol></nav>
    <main className={`main-card step-${step}`} aria-busy={busy}>
      <div className="section-top page-heading"><h2 ref={heading} tabIndex={-1}>{steps[step - 1].label}</h2><span className="muted">STEP {String(step).padStart(2, '0')} / 05</span></div>
      {error && <div className="error" role="alert">{error}</div>}
      {busy && <div className="loading" role="status"><span className="spinner" />{step === 4 ? '블로그 글을 만들고 있습니다. 잠시만 기다려주세요.' : '아이디어를 정리하고 있습니다. 잠시만 기다려주세요.'}</div>}
      {step === 1 && <section className="panel input-panel"><span className="eyebrow">RIFIT BLOG WRITING</span><h3>어떤 이야기를 나누고 싶나요?</h3><p className="muted input-description">떠오른 아이디어를 자유롭게 입력해 주세요.<br />키워드만 있어도 괜찮아요.</p><label htmlFor="source">아이디어 입력</label><textarea id="source" value={source} disabled={busy} onChange={e => setSource(e.target.value)} placeholder={'블로그에 쓰고 싶은 내용이나 주제를 자유롭게 적어주세요.\n예) 계절이 바뀌면서 안 입는 옷을 정리하고 싶어요.'} rows={5} /><div className="input-bottom"><span className="muted character-count">{source.length.toLocaleString()}자</span><button className="primary" disabled={busy || !source.trim()} onClick={() => run(async () => { const data = await expandIdea(source); setCandidates(data.candidates); setSelected(null); setIdea(null); setResult(null); go(2); })}>아이디어 확장하기 →</button></div></section>}
      {step === 2 && <CandidateSelection candidates={candidates} selectedId={selected?.candidate_id} mode={generationMode} busy={busy} onSelect={candidate => { if (selected?.candidate_id !== candidate.candidate_id) { setRevision(''); setIdea(null); setResult(null); } setSelected(candidate); go(3); }} onGenerateLive={() => run(async () => { const data = await requestExpansion(source, setGenerationMode, true); setCandidates(data.candidates); setSelected(null); setRevision(''); setIdea(null); setResult(null); })} />}
      {step === 3 && selected && <section className="panel input-panel"><span className="eyebrow">선택한 아이디어</span><h3>{selected.title}</h3><p className="question">{selected.key_question}</p><label htmlFor="revision">추가 수정 요청 <span className="muted">선택</span></label><textarea id="revision" rows={5} value={revision} disabled={busy} onChange={e => setRevision(e.target.value)} placeholder={'추가로 원하는 방향이 있다면 적어주세요.\n예: 환경 이야기는 줄이고 옷장 정리에 집중해주세요.'}/><div className="actions"><button className="primary" disabled={busy} onClick={() => run(async () => { setIdea(await refineIdea(source, selected, revision)); setResult(null); go(4); })}>아이디어 확정하기 →</button></div></section>}
      {step === 4 && idea && <section className="panel input-panel"><span className="eyebrow">집필 전 마지막 확인</span><h3>{idea.title}</h3><p>{idea.summary}</p><dl className="idea-facts">{[['키워드', idea.keyword], ['형식', idea.content_format], ['관점', idea.content_perspective], ['핵심 질문', idea.key_question], ['독자', idea.target_reader]].map(([label, value]) => value && <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>{!!idea.outline?.length && <><h4>글의 흐름</h4><ol className="outline">{idea.outline.map((line, i) => <li key={i}>{line}</li>)}</ol></>}<div className="actions"><button disabled={busy} onClick={() => go(3)}>방향 다시 다듬기</button><button className="primary" disabled={busy} onClick={() => run(async () => { setResult(await generateWorkflow(source, idea)); go(5); })}>이 내용으로 글 생성 →</button></div></section>}
      {step === 5 && result && <ResultView result={result}/>}
      {step > 1 && <footer className="navigation"><div><button disabled={busy} onClick={() => go(previousStep[step])}>← 이전 단계</button>{step > 2 && <button disabled={busy} onClick={() => go(2)}>← 후보 선택으로</button>}<button disabled={busy} onClick={() => go(1)}>원본 메모 보기</button></div><button disabled={busy} onClick={reset}>새 아이디어 작성</button></footer>}
    </main><footer className="footnote">RIFIT BLOG WORKSPACE <span>발행 전, 초안의 사실과 표현을 직접 확인해주세요.</span></footer>
  </div>;
}
