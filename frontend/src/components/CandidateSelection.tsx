import type { IdeaCandidate } from '../types/models';
import type { GenerationMode } from '../services/api';

type Props = {
  candidates: IdeaCandidate[];
  selectedId?: string;
  mode: GenerationMode;
  busy: boolean;
  onSelect: (candidate: IdeaCandidate) => void;
  onGenerateLive: () => void;
};

export default function CandidateSelection({ candidates, selectedId, mode, busy, onSelect, onGenerateLive }: Props) {
  return <section className="candidate-selection">
    <div className="candidate-heading"><span className="eyebrow">CHOOSE YOUR STORY</span>
      <h3>하나의 주제, 세 가지 글의 방향</h3>
      <p className="muted">제목과 소제목, 구체적인 전개 방향을 비교하고 쓰고 싶은 글을 선택하세요.</p>
    </div>
    <div className="candidate-generation">
      <div><strong>{mode === 'mock' ? '현재는 화면 확인용 샘플입니다' : '다른 아이디어가 필요하신가요?'}</strong>
        <p className="muted">아래 버튼을 누를 때만 OpenAI로 후보를 생성합니다. API 비용이 발생하며, 검증 보정 포함 최대 2회 호출합니다. 글 생성은 실행하지 않습니다.</p>
        {mode === 'mock' && <p className="muted">이번 요청의 아이디어만 실제 AI로 생성하며, 이후 단계의 서버 모드는 바뀌지 않습니다.</p>}
      </div>
      <button className="primary" disabled={busy} onClick={onGenerateLive}>{busy ? '후보를 만들고 있어요…' : '실제 AI로 후보 3개 생성'}</button>
    </div>
    <div className="candidate-grid">{candidates.map((candidate, index) =>
      <article className={`panel candidate ${selectedId === candidate.candidate_id ? 'selected' : ''}`} key={candidate.candidate_id}>
        <div className="candidate-meta"><span className="eyebrow">IDEA {String(index + 1).padStart(2, '0')}</span><span>{mode === 'mock' ? '샘플' : candidate.content_format}</span></div>
        <h3>{candidate.title}</h3>
        <p className="candidate-subtitle">{candidate.perspective}</p>
        <div className="candidate-direction"><h4>이 글의 방향</h4><p>{candidate.brief_description}</p></div>
        <button aria-pressed={selectedId === candidate.candidate_id} disabled={busy} className="select-button" onClick={() => onSelect(candidate)}>이 아이디어 선택 →</button>
      </article>
    )}</div>
    {!candidates.length && <p>표시할 후보가 없습니다. 입력 화면에서 다시 시도해주세요.</p>}
  </section>;
}
