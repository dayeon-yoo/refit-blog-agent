import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import type { WorkflowResult } from '../types/models';

export default function ResultView({ result }: { result: WorkflowResult }) {
  const [notice, setNotice] = useState('');
  const { post, qc, tags, image_plan } = result;
  async function copy(text: string, label: string) {
    try { await navigator.clipboard.writeText(text); setNotice(`${label}을 복사했습니다.`); }
    catch { setNotice('복사하지 못했습니다. 브라우저의 클립보드 권한을 확인해주세요.'); }
  }
  const review = (
    <section className={`qc panel ${qc.status === 'PASS' ? 'pass' : 'review'}`} aria-label="최종 검토">
      <span className="eyebrow">최종 검토</span>
      <h2>{qc.status === 'PASS' ? '검토 통과' : qc.status === 'BLOCK' ? '수정 필요' : '검토 권장'}</h2>
      <p>{qc.summary || '아래 본문을 직접 확인한 뒤 사용해주세요.'}</p>
      {qc.issues.map((issue, i) => <div className="issue" key={i}>
        <strong>{issue.severity === 'error' ? '수정 필요' : '확인 권장'} · {issue.message}</strong>
        {issue.evidence && <blockquote>{issue.evidence}</blockquote>}
        {issue.suggestion && <p>{issue.suggestion}</p>}
      </div>)}
    </section>
  );
  return <>
    <p role="status" className="copy-notice">{notice}</p>
    <div className="result-grid">
      <article className="panel article">
        <div className="section-top"><span className="eyebrow">블로그 초안</span><button onClick={() => copy([post.title, post.content, post.cta].filter(Boolean).join('\n\n'), '본문')}>본문 복사</button></div>
        <h1>{post.title}</h1>
        <div className="prose"><ReactMarkdown skipHtml>{post.content}</ReactMarkdown></div>
        {post.cta && <div className="closing"><ReactMarkdown skipHtml>{post.cta}</ReactMarkdown></div>}
      </article>
      <div className="result-aside">
        <section className="panel"><div className="section-top"><h2>태그</h2><button disabled={!tags.length} onClick={() => copy(tags.map(tag => `#${tag.replace(/^#+/, '')}`).join(' '), '태그')}>태그 복사</button></div>
          <div className="chips">{tags.map((tag, i) => <span key={i}>#{tag.replace(/^#+/, '')}</span>)}</div>
          {!tags.length && <p>추천 태그가 없습니다.</p>}
        </section>
        <section className="panel"><span className="eyebrow">이미지 제작 가이드</span><h2>이런 장면을 준비해보세요</h2><p className="muted">실제 이미지가 아닌 이미지 계획입니다.</p>
          {image_plan?.images.map((item, i) => <div className="image-item" key={i}>
            <span className="number">{String(i + 1).padStart(2, '0')}</span><h3>{item.placement}</h3><p>{item.alt_text || item.purpose}</p><p className="muted">{item.purpose}</p>
            <details><summary>이미지 구상 자세히</summary><p>{item.prompt}</p></details>
          </div>)}
          {!image_plan?.images.length && <p>이미지 계획이 없습니다.</p>}
        </section>
      </div>
    </div>
    {review}
  </>;
}
